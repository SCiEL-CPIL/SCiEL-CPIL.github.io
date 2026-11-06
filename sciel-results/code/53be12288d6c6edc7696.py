"""Block-structured construction + simulated annealing over block patterns."""
import numpy as np
import random
import time

N = 256
BLOCK_SIZE = 16
NUM_BLOCKS = N // BLOCK_SIZE  # 16

# Precompute all arithmetic progressions and their block incidence counts
# For each AP, count how many positions fall in each block
# ap_block_counts[ap_idx, block_idx] = number of positions in that AP within block_idx

def _build_ap_block_counts():
    """For each AP, compute count of positions in each block."""
    n_aps = 0
    # First pass: count APs
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            length = (N - start) // step + 1
            if length >= 1:
                n_aps += 1
    
    counts = np.zeros((n_aps, NUM_BLOCKS), dtype=np.int8)
    ap_idx = 0
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            length = (N - start) // step + 1
            if length < 1:
                continue
            # Count positions per block
            for pos in range(start - 1, N, step):  # 0-indexed
                block_idx = pos // BLOCK_SIZE
                counts[ap_idx, block_idx] += 1
            ap_idx += 1
    return counts

# Precompute block counts at module load
_AP_BLOCK_COUNTS = _build_ap_block_counts()

# Library of candidate 16-sign patterns
# We'll use several families:
# 1. All +1, all -1
# 2. Alternating +1,-1 (and shifts)
# 3. Runs of 4: ++++----++++----, etc.
# 4. Random low-discrepancy patterns
# 5. Walsh-like patterns

def _generate_block_library():
    """Generate a diverse library of 16-element sign patterns."""
    patterns = []
    seen = set()
    
    def add(p):
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            patterns.append(np.array(p, dtype=np.int8))
            # Also add the negation
            neg = tuple(-x for x in p)
            if neg not in seen:
                seen.add(neg)
                patterns.append(np.array(neg, dtype=np.int8))
    
    # All ones / all minus
    add([1] * 16)
    add([-1] * 16)
    
    # Alternating
    add([1 if i % 2 == 0 else -1 for i in range(16)])
    add([-1 if i % 2 == 0 else 1 for i in range(16)])
    
    # Runs of 4
    for phase in range(4):
        p = [1 if ((i + phase) // 4) % 2 == 0 else -1 for i in range(16)]
        add(p)
    
    # Runs of 2
    for phase in range(2):
        p = [1 if ((i + phase) // 2) % 2 == 0 else -1 for i in range(16)]
        add(p)
    
    # Runs of 8
    for phase in range(2):
        p = [1 if ((i + phase) // 8) % 2 == 0 else -1 for i in range(16)]
        add(p)
    
    # Walsh-Hadamard-like (Rademacher functions)
    # f_k(i) = (-1)^(popcount(i & k)) for k in 0..15
    for k in range(16):
        p = []
        for i in range(16):
            # Count set bits in (i & k)
            v = i & k
            pc = bin(v).count('1')
            p.append(1 if pc % 2 == 0 else -1)
        add(p)
    
    # Some random low-discrepancy patterns
    rng = random.Random(42)
    for _ in range(64):
        p = [rng.choice((-1, 1)) for _ in range(16)]
        add(p)
    
    # Patterns optimized for small APs within a block (local low discrepancy)
    # Greedy: assign signs to minimize max AP sum within the 16 positions
    for seed in range(20):
        rng2 = random.Random(100 + seed)
        p = [0] * 16
        for i in range(16):
            # Try both signs, pick the one that minimizes worst local AP sum
            best_val = None
            best_sign = 1
            for s in (1, -1):
                p[i] = s
                # Compute max abs sum over all APs within positions 0..i
                max_s = 0
                for st in range(1, i + 1):
                    for st2 in range(0, i + 1):
                        if st2 + st > i:
                            break
                        ssum = sum(p[st2:st2 + st + 1])
                        max_s = max(max_s, abs(ssum))
                if best_val is None or max_s < best_val:
                    best_val = max_s
                    best_sign = s
            p[i] = best_sign
        add(p)
    
    return patterns

_BLOCK_LIBRARY = _generate_block_library()
NUM_PATTERNS = len(_BLOCK_LIBRARY)

# Precompute: for each pattern p and each AP, the contribution = sum of pattern * block_count
# pattern_ap_contrib[pattern_idx, ap_idx] = sum over blocks of pattern[block] * ap_block_counts[ap, block]
_PATTERN_AP_CONTRIB = np.zeros((NUM_PATTERNS, _AP_BLOCK_COUNTS.shape[0]), dtype=np.int16)
for pi, pat in enumerate(_BLOCK_LIBRARY):
    _PATTERN_AP_CONTRIB[pi] = _AP_BLOCK_COUNTS @ pat  # shape (num_aps,)


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_score = np.inf
    best_assign = None
    num_aps = _AP_BLOCK_COUNTS.shape[0]
    
    # Simulated annealing over block pattern choices
    # State: block_choice[0..15] = index into _BLOCK_LIBRARY
    # Objective: max over APs of |sum over blocks of pattern[block][pos_in_block] * ... |
    # Wait - this isn't quite right. The AP sum depends on WHICH positions in the block are hit,
    # not just the count. Let me reconsider.
    
    # Actually, the AP sum for a given AP = sum over positions in AP of sign[position].
    # If we group by block: AP sum = sum over blocks b of (sum over positions in AP ∩ block b of sign[pos]).
    # The inner sum depends on the specific pattern AND which positions in the block are in the AP.
    # So we can't just use block counts; we need the exact pattern values at the specific positions.
    
    # Let me redo: precompute for each AP and each block, the list of local indices (0-15) within the block
    # that are part of the AP. Then for a given pattern, the contribution from that block is
    # sum of pattern[local_idx] for local_idx in the list.
    
    # This is more complex. Let me build a proper structure.
    # For each AP, for each block, store the list of local indices.
    # Then pattern_ap_contrib[pattern, ap] = sum over blocks of sum(pattern[local_idx] for local_idx in ap_block_local_indices[ap, block])
    
    # Let me rebuild this properly.
    
    # Rebuild: for each AP, for each block, the list of local indices
    ap_block_locals = []  # list of lists: ap_block_locals[ap][block] = list of local indices
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            length = (N - start) // step + 1
            if length < 1:
                continue
            block_locals = [[] for _ in range(NUM_BLOCKS)]
            for pos in range(start - 1, N, step):
                block_idx = pos // BLOCK_SIZE
                local_idx = pos % BLOCK_SIZE
                block_locals[block_idx].append(local_idx)
            ap_block_locals.append(block_locals)
    
    # Now compute pattern_ap_contrib properly
    # pattern_ap_contrib[pattern_idx, ap_idx] = sum over blocks of sum(pattern[li] for li in ap_block_locals[ap][block])
    contrib = np.zeros((NUM_PATTERNS, num_aps), dtype=np.int16)
    for pi, pat in enumerate(_BLOCK_LIBRARY):
        for ai, block_locals in enumerate(ap_block_locals):
            s = 0
            for b in range(NUM_BLOCKS):
                for li in block_locals[b]:
                    s += pat[li]
            contrib[pi, ai] = s
    
    # Now the objective for a block assignment (choice[0..15]):
    # For each AP: total_sum = sum over blocks b of contrib[choice[b], ap_idx]
    # Score = max over APs of |total_sum|
    
    # This is a 16-variable discrete optimization. Let's use simulated annealing.
    
    def evaluate(choice):
        # choice: array of shape (NUM_BLOCKS,) with values in [0, NUM_PATTERNS)
        # total_sums[ap] = sum over blocks of contrib[choice[block], ap]
        total_sums = np.zeros(num_aps, dtype=np.int16)
        for b in range(NUM_BLOCKS):
            total_sums += contrib[choice[b]]
        return int(np.max(np.abs(total_sums)))
    
    # Initial: random assignment
    rng = random.Random(12345)
    current_choice = np.array([rng.randint(0, NUM_PATTERNS - 1) for _ in range(NUM_BLOCKS)], dtype=np.int32)
    current_score = evaluate(current_choice)
    best_choice = current_choice.copy()
    best_score = current_score
    
    # Simulated annealing
    T = 5.0
    cooling = 0.995
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        # Pick a random block to change
        b = rng.randint(0, NUM_BLOCKS - 1)
        old_pat = current_choice[b]
        new_pat = rng.randint(0, NUM_PATTERNS - 1)
        if new_pat == old_pat:
            continue
        
        current_choice[b] = new_pat
        new_score = evaluate(current_choice)
        
        delta = new_score - current_score
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_choice = current_choice.copy()
        else:
            current_choice[b] = old_pat
        
        T *= cooling
    
    # Build the final sign assignment
    signs = np.zeros(N, dtype=np.int8)
    for b in range(NUM_BLOCKS):
        pat = _BLOCK_LIBRARY[best_choice[b]]
        signs[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE] = pat
    
    return [int(x) for x in signs]
