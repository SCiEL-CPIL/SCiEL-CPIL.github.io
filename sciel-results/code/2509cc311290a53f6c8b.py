"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Phase 0: Build initial superpermutation using recursive construction
    # For n=7, the classic construction gives length 5913
    word = _build_superpermutation(n)
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Phase 1: Build coverage counts
    def build_coverage(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts
    
    improved = list(word)
    counts = build_coverage(improved, n)
    
    total_perms = math.factorial(n)
    
    def get_missing(counts):
        return total_perms - len(counts)
    
    # Phase 2: Aggressive greedy deletion with multiple passes
    # Key insight: A position can be deleted if all n permutations containing it
    # have count > 1 (i.e., they appear elsewhere)
    
    def can_delete(pos, w, counts, n):
        """Check if position pos can be deleted without losing any permutation."""
        j_min = max(0, pos - n + 1)
        j_max = min(pos, len(w) - n)
        for j in range(j_min, j_max + 1):
            if counts[tuple(w[j:j+n])] <= 1:
                return False
        return True
    
    def delete_pos(pos, w, counts, n):
        """Delete position pos, updating counts."""
        j_min = max(0, pos - n + 1)
        j_max = min(pos, len(w) - n)
        affected = []
        for j in range(j_min, j_max + 1):
            perm = tuple(w[j:j+n])
            affected.append(perm)
            counts[perm] -= 1
            if counts[perm] == 0:
                del counts[perm]
        w.pop(pos)
    
    def greedy_pass(w, counts, n, order):
        """One pass of greedy deletion. Returns number of deletions."""
        deletions = 0
        if order == "fwd":
            i = 0
            while i < len(w):
                if can_delete(i, w, counts, n):
                    delete_pos(i, w, counts, n)
                    deletions += 1
                else:
                    i += 1
        elif order == "bwd":
            i = len(w) - 1
            while i >= 0:
                if can_delete(i, w, counts, n):
                    delete_pos(i, w, counts, n)
                    deletions += 1
                else:
                    i -= 1
        elif order == "random":
            # Try random positions, multiple rounds
            rounds = 3
            for _ in range(rounds):
                if len(w) == 0:
                    break
                positions = list(range(len(w)))
                random.shuffle(positions)
                for pos in positions:
                    if 0 <= pos < len(w) and can_delete(pos, w, counts, n):
                        delete_pos(pos, w, counts, n)
                        deletions += 1
        return deletions
    
    # Run multiple greedy passes until no more deletions possible
    while time.time() < deadline - 30:
        total_deleted = 0
        for order in ["fwd", "bwd", "random", "fwd", "bwd"]:
            if time.time() > deadline - 30:
                break
            d = greedy_pass(improved, counts, n, order)
            total_deleted += d
        if total_deleted == 0:
            break
    
    if time.time() > deadline:
        return "".join(map(str, improved))
    
    # Phase 3: Targeted deletion - look for positions where the character 
    # appears in many redundant permutations
    # This is a heuristic: prefer deleting positions where the surrounding 
    # context is most "common"
    
    def targeted_deletion(w, counts, n, max_attempts=5000):
        """Try to find deletions by examining local context frequency."""
        deletions = 0
        attempts = 0
        while attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1
            if len(w) <= n:
                break
            pos = random.randint(0, len(w) - 1)
            if can_delete(pos, w, counts, n):
                delete_pos(pos, w, counts, n)
                deletions += 1
        return deletions
    
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        d = targeted_deletion(improved, counts, n)
        if d == 0:
            break
    
    if time.time() > deadline:
        return "".join(map(str, improved))
    
    # Phase 4: Local search with swaps and moves
    # Try to improve by moving characters to create more overlap opportunities
    
    def try_swap(w, counts, n, i, j):
        """Try swapping positions i and j. Returns True if valid and applied."""
        if i == j:
            return False
        w[i], w[j] = w[j], w[i]
        # Rebuild counts (expensive but necessary for correctness)
        new_counts = build_coverage(w, n)
        if get_missing(new_counts) == 0:
            counts.clear()
            counts.update(new_counts)
            return True
        else:
            w[i], w[j] = w[j], w[i]
            return False
    
    def try_move(w, counts, n, i, j):
        """Try moving character at i to position j. Returns True if valid and applied."""
        if i == j:
            return False
        char = w.pop(i)
        if j > i:
            j -= 1
        w.insert(j, char)
        new_counts = build_coverage(w, n)
        if get_missing(new_counts) == 0:
            counts.clear()
            counts.update(new_counts)
            return True
        else:
            w.pop(j)
            w.insert(i, char)
            return False
    
    # Run local search
    local_search_steps = 0
    while time.time() < deadline - 10 and local_search_steps < 50000:
        local_search_steps += 1
        if len(improved) < 3:
            break
        
        # Randomly choose operation
        op = random.random()
        if op < 0.3:
            # Swap adjacent
            i = random.randint(0, len(improved) - 2)
            try_swap(improved, counts, n, i, i+1)
        elif op < 0.5:
            # Swap distant
            i = random.randint(0, len(improved) - 1)
            j = random.randint(0, len(improved) - 1)
            try_swap(improved, counts, n, i, j)
        elif op < 0.8:
            # Move to adjacent
            i = random.randint(0, len(improved) - 1)
            j = random.choice([i-1, i+1])
            if 0 <= j < len(improved):
                try_move(improved, counts, n, i, j)
        else:
            # Move to random position
            i = random.randint(0, len(improved) - 1)
            j = random.randint(0, len(improved) - 1)
            try_move(improved, counts, n, i, j)
        
        # Periodically run greedy pass to clean up
        if local_search_steps % 500 == 0:
            greedy_pass(improved, counts, n, "fwd")
            greedy_pass(improved, counts, n, "bwd")
    
    if time.time() > deadline:
        return "".join(map(str, improved))
    
    # Phase 5: Final greedy cleanup
    for order in ["fwd", "bwd", "random", "fwd", "bwd", "random"]:
        if time.time() > deadline - 2:
            break
        greedy_pass(improved, counts, n, order)
    
    # Final validation
    final_counts = build_coverage(improved, n)
    if get_missing(final_counts) > 0:
        # This shouldn't happen, but fallback
        improved = list(word)
    
    return "".join(map(str, improved))

def _build_superpermutation(n):
    """Build a superpermutation using the classic recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    
    # Start with superpermutation of 1..n-1
    word = _build_superpermutation(n - 1)
    
    # Find all (n-1)-length windows that are permutations of 1..n-1
    seen = set()
    perms = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Build new word by inserting n between each permutation
    new_word = []
    for p in perms:
        # The block is: p + [n] + p
        # But we need to handle overlap with existing new_word
        block = list(p) + [n] + list(p)
        
        # Find maximum overlap with end of new_word
        max_overlap = min(len(new_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if new_word[len(new_word) - o:] == block[:o]:
                overlap = o
                break
        
        new_word.extend(block[overlap:])
    
    return new_word
# EVOLVE-BLOCK-END
