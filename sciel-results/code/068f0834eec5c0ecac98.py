import time
import numpy as np

N = 16
MAX_TIME = 55.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    This is the standard 63-comparator network.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1_s = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            
            return step1_s + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # 1. Generate Initial Network (Batcher 63)
    initial_comps = _generate_batcher_network()
    
    # Verify
    if not _is_sorted_network_vectorized(initial_comps, bits):
        # Fallback to Odd-Even Transposition (120) if Batcher is broken (should not happen)
        initial_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    initial_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    initial_comps.append((j, j + 1))
    
    best_comps = initial_comps[:]
    best_len = len(best_comps)
    
    # Helper to check validity
    def check(comps):
        return _is_sorted_network_vectorized(comps, bits)
    
    # 2. Aggressive Multi-Phase Reduction
    
    # Phase A: Random Removal (Forward Greedy)
    rng = np.random.default_rng(123)
    
    # Try multiple random orders
    for attempt in range(3):
        if time.time() > deadline - 10.0:
            break
        current_comps = best_comps[:]
        order = rng.permutation(len(current_comps))
        for idx in order:
            if time.time() > deadline - 10.0:
                break
            if len(current_comps) <= idx:
                continue
            removed = current_comps.pop(idx)
            if check(current_comps):
                pass # Keep removed
            else:
                current_comps.insert(idx, removed)
        
        if len(current_comps) < len(best_comps):
            best_comps = current_comps
            best_len = len(best_comps)
    
    # Phase B: Adjacent Pair Removal (Structural)
    # Try to remove pairs of adjacent comparators if possible
    for attempt in range(2):
        if time.time() > deadline - 5.0:
            break
        current_comps = best_comps[:]
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Random shuffle of pair positions
            possible_pairs = list(range(len(current_comps) - 1))
            rng.shuffle(possible_pairs)
            for idx in possible_pairs:
                if time.time() > deadline - 5.0:
                    break
                # Remove pair at idx and idx+1
                comp1 = current_comps[idx]
                comp2 = current_comps[idx+1]
                # Check if they are identical or adjacent in a way that might be redundant
                # Just try removing both
                temp = current_comps[:idx] + current_comps[idx+2:]
                if check(temp):
                    current_comps = temp
                    changed = True
                    break # Restart scanning as indices changed
        
        if len(current_comps) < len(best_comps):
            best_comps = current_comps
            best_len = len(best_comps)
    
    # Phase C: Local Search with Swaps (Refinement)
    # Try swapping comparators to enable further removals
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline - 2.0:
        current_comps = best_comps[:]
        improved = False
        len_cur = len(current_comps)
        
        if len_cur > 60: # Only bother if we can get below 60? Or just keep going?
            # Try random removal
            idx = rng.integers(0, len_cur)
            removed = current_comps.pop(idx)
            if check(current_comps):
                best_comps = current_comps
                best_len -= 1
                improved = True
            else:
                current_comps.insert(idx, removed)
        
        if not improved:
            # Try random swap
            idx = rng.integers(0, len_cur)
            old_comp = current_comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp != old_comp:
                current_comps[idx] = new_comp
                if check(current_comps):
                    # If swap works, try removing something immediately?
                    # Or just accept the swap? A swap doesn't reduce count.
                    # But it might enable a removal.
                    # Let's just accept valid swaps to explore landscape?
                    # Actually, standard local search for min comparators usually
                    # only accepts moves that reduce count. Swaps are neutral.
                    # So we should only keep swap if it allows a subsequent removal.
                    # This is a 2-move local search.
                    
                    # Try to remove a random comparator after the swap
                    temp_comps = current_comps[:]
                    rem_idx = rng.integers(0, len(temp_comps))
                    temp_rem = temp_comps.pop(rem_idx)
                    if check(temp_comps):
                        best_comps = temp_comps
                        best_len -= 1
                        improved = True
                    # If not, discard the whole 2-move sequence (swap+remove)
        
        if not improved and len_cur <= 60:
            # We might be stuck. If we are at 60, that's the target.
            # If we are above 60 and stuck for a while, we can break.
            # But let's just keep trying until time runs out to be safe.
            pass

    return best_comps

if __name__ == "__main__":
    comps = construct()
    print(f"Constructors: {len(comps)}")
