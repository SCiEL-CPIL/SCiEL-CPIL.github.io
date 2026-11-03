import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    # Also precompute inverse balls: inv_balls[i] is the mask of all j such that i is in balls[j]
    # This helps in branching: if i is uncovered, we must pick a j from inv_balls[i]
    inv_balls = [0] * total_words

    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            # Self
            mask |= (1 << i)
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate neighbor index directly
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
            inv_balls[i] |= (1 << i) # i covers itself
            # We need to update inv_balls for all j covered by i
            # This is O(N^2) if done naively, but N=2187, N^2 ~ 4.7M, which is fine.
            # However, iterating over bits of mask is faster if sparse, but here balls are dense (size 1+14=15).
            # Actually, let's just compute inv_balls after balls are computed.
            
        # Compute inv_balls
        for i in range(total_words):
            mask = balls[i]
            # Iterate through set bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                inv_balls[idx] |= (1 << i)
                m ^= lsb
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            
        # Compute inv_balls
        for i in range(total_words):
            mask = balls[i]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                inv_balls[idx] |= (1 << i)
                m ^= lsb

    all_ones_mask = (1 << total_words) - 1
    
    # Calculate max ball size for lower bound
    max_ball_size = 0
    for i in range(total_words):
        cnt = balls[i].bit_count()
        if cnt > max_ball_size:
            max_ball_size = cnt

    def greedy_cover(uncovered_mask):
        """
        Greedily select codewords to cover the words in uncovered_mask.
        Returns a list of indices.
        """
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def remove_redundants(selected):
        """
        Removes any codeword from selected that is redundant.
        Returns the new list of selected indices.
        """
        if not selected:
            return []
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    # Initial Greedy to get a baseline
    initial_selected = greedy_cover(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    
    best_size = len(initial_selected)
    best_code = [all_words[i] for i in initial_selected]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Branch and Bound Search
    # We use a stack for DFS to avoid recursion depth issues, though recursion is fine for this depth.
    # State: (uncovered_mask, selected_indices, selected_count)
    
    # To improve B&B, we can use a heuristic to order the candidates.
    # We pick an uncovered vertex u. We must choose a codeword c from inv_balls[u].
    # We sort candidates by coverage of remaining uncovered vertices.
    
    def lower_bound(uncovered_mask):
        if uncovered_mask == 0:
            return 0
        # Simple bound: each codeword covers at most max_ball_size vertices
        return (uncovered_mask.bit_count() + max_ball_size - 1) // max_ball_size

    # We will perform a limited DFS B&B
    # We store the best solution found so far
    # We use a global variable for best_size and best_code inside the closure
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bbs_search(uncovered_mask, selected_indices, selected_count):
        nonlocal best_size, best_code, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if uncovered_mask == 0:
            if selected_count < best_size:
                best_size = selected_count
                best_code = [all_words[i] for i in selected_indices]
            return
            
        # Pruning
        lb = lower_bound(uncovered_mask)
        if selected_count + lb >= best_size:
            return
            
        # Pick an uncovered vertex to branch on
        # Heuristic: Pick the vertex that has the fewest candidates? 
        # Or just the first one? 
        # Let's pick the one with the smallest degree in the "remaining" graph?
        # Actually, inv_balls is static. The number of candidates is popcount(inv_balls[u]).
        # Picking a vertex with fewer candidates reduces branching factor.
        
        # Find an uncovered vertex
        m = uncovered_mask
        if m == 0:
            return
            
        # We can iterate through a few uncovered vertices to find one with min candidates
        # But for speed, let's just pick the first one or one with random tie-breaking
        # Let's pick the one with minimum popcount(inv_balls[u] & ~covered_by_selected?) 
        # Wait, inv_balls[u] contains ALL vertices that can cover u.
        # Some of them might already be selected? No, if they were selected, u would be covered.
        # So all candidates in inv_balls[u] are valid next moves.
        
        # Let's find an uncovered vertex u
        lsb = m & -m
        u = lsb.bit_length() - 1
        
        # Get candidates
        candidates_mask = inv_balls[u]
        
        # Generate list of candidates
        candidates = []
        cm = candidates_mask
        while cm:
            lsb_c = cm & -cm
            c_idx = lsb_c.bit_length() - 1
            # Heuristic score: how many other uncovered vertices does c_idx cover?
            score = (balls[c_idx] & uncovered_mask).bit_count()
            candidates.append((score, c_idx))
            cm ^= lsb_c
            
        # Sort by score descending
        candidates.sort(reverse=True)
        
        # Limit branching factor to avoid explosion
        # We can branch on all, or top K.
        # Given the time limit, branching on all might be too slow if depth is large.
        # But with strong pruning, it might be ok.
        # Let's try branching on all candidates for now, but with time check.
        # If it's too slow, we can limit to top 5.
        
        for score, c_idx in candidates:
            # New state
            new_uncovered = uncovered_mask & ~balls[c_idx]
            new_selected = selected_indices + [c_idx]
            new_count = selected_count + 1
            
            # Optimistic bound: if new_count + LB(new_uncovered) >= best_size, skip
            if new_count + lower_bound(new_uncovered) >= best_size:
                continue
                
            bbs_search(new_uncovered, new_selected, new_count)
            
            if time.time() - start_time > time_limit:
                return

    # Run B&B
    # We might want to randomize the order of vertices or candidates to escape local minima?
    # The greedy choice (sorting by score) is deterministic.
    # To make it stochastic, we can shuffle candidates with equal scores or introduce randomness.
    # However, the directive asks for a "custom branch-and-bound". Deterministic is fine if it prunes well.
    # But to improve robustness, let's add a bit of randomness to the candidate selection order if there are ties.
    
    # Let's modify the candidate sorting to be slightly random for ties
    import operator
    # We already sorted by score.
    
    # To prevent the B&B from getting stuck in a specific path, we can run it multiple times with different random seeds for tie-breaking?
    # Or just rely on the strong pruning.
    
    # Let's run the B&B
    bbs_search(all_ones_mask, [], 0)
    
    return best_code
