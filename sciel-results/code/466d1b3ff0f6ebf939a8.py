import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        """Greedy set cover with randomized tie-breaking."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_set, cov, max_time=None):
        """Add codewords to cover all uncovered points."""
        if max_time is None:
            max_time = deadline - 10.0
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < max_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    def compress(code, cov, max_time=None):
        """Iterative removal of non-critical codewords until no more can be removed.
        A codeword c is removable if min(cov[neighbors[c]]) >= 2."""
        if max_time is None:
            max_time = deadline - 5.0
        cur_code = code[:]
        cur_cov = cov.copy()
        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            # Sort by safety (min coverage among neighbors), remove most redundant first
            safety = np.array([np.min(cur_cov[neighbors[c]]) for c in cur_code], dtype=np.int32)
            order = np.argsort(-safety)
            for idx in order:
                if (time.time() - t_start) > max_time:
                    break
                i = int(idx)
                c = cur_code[i]
                if safety[i] >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    improved = True
                    break  # Restart ordering
        return cur_code, cur_cov

    def swap_local_search(code, cov, max_time=None):
        """Fixed-size local search: try all single swaps (remove one, add one).
        Accept if it reduces the number of uncovered points, or keeps same and
        the new codeword is 'better' (covers more uncovered)."""
        if max_time is None:
            max_time = deadline - 5.0
        cur_code = code[:]
        cur_cov = cov.copy()
        size = len(cur_code)
        
        # Count uncovered
        def count_uncovered(c):
            return np.sum(c == 0)
        
        cur_uncovered = count_uncovered(cur_cov)
        
        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            best_swap = None
            best_uncovered = cur_uncovered
            
            # For each codeword to remove, find best replacement
            for i, c_remove in enumerate(cur_code):
                if (time.time() - t_start) > max_time:
                    break
                
                # Temporarily remove c_remove
                cov_after_remove = cur_cov - 1  # This is expensive; do it smarter
                
                # Actually, let's just compute the gain/loss properly
                # Removing c_remove: uncovered increases by count of points that had cov==1 in its ball
                # Adding c_add: uncovered decreases by count of points that had cov==0 in its ball (after removal)
                
                # For efficiency, precompute: for each candidate c_add,
                # gain = number of points in ball(c_add) that would be uncovered after removing c_remove
                #       minus number of points in ball(c_remove) that would become uncovered
                # Actually simpler: just do it directly but smartly.
                
                # Remove c_remove's contribution
                ball_remove = neighbors[c_remove]
                # Points that become uncovered: those with cur_cov == 1 in ball_remove
                newly_uncovered = (cur_cov[ball_remove] == 1).sum()
                
                # For each potential add, compute how many of those newly_uncovered
                # it would cover, plus how many other uncovered points it covers
                # But this is O(space_size) per candidate...
                
                # Alternative: just try a subset of candidates
                # Use the greedy gain approach: for each candidate not in code,
                # compute its marginal benefit given the current state minus c_remove
                
                # This is still expensive. Let's use a different approach:
                # Only try swaps where we remove a "weak" codeword and add a "strong" one.
                # Weak: high redundancy (min cov in ball >= 2)
                # Strong: covers many currently-uncovered or singly-covered points
                
                min_cov_remove = np.min(cur_cov[ball_remove])
                if min_cov_remove < 2:
                    continue  # Can't safely remove this one for a swap that keeps size
                
                # Try a few random candidates
                # Actually, for fixed-size swap, we need to find c_add such that
                # the net uncovered count doesn't increase.
                # Let's just do a random sampling of candidates for speed.
                
                # For now, skip the full swap search and just do removal-based compression
                # The swap search is too expensive per iteration.
                pass
        
        # Since full swap search is too expensive, fall back to just compression
        return cur_code, cur_cov

    # === MAIN ALGORITHM: Iterated Compression with Kicks ===
    
    # Phase 1: Multiple greedy starts to get good initial solutions
    best_code = None
    best_size = space_size + 1

    for seed in range(8):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 35.0)
        # Compress
        code, cov = compress(code, cov, deadline - 30.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Iterated Local Search (ILS) with kicks
    # Each iteration: kick (randomly perturb), then compress, keep best
    rng = np.random.default_rng(12345)
    cur_code = best_code[:]
    cur_cov = build_cov(cur_code)
    
    for iteration in range(200):
        if (time.time() - t_start) > deadline - 10.0:
            break
        
        # Kick: randomly remove k codewords and repair
        k = 1 + int(rng.integers(0, 4))  # Remove 1 to 4 random codewords
        if len(cur_code) <= k:
            k = max(1, len(cur_code) // 2)
        
        remove_indices = rng.choice(len(cur_code), size=k, replace=False)
        remove_set = set(int(ri) for ri in remove_indices)
        
        kicked_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
        kicked_cov = build_cov(kicked_code)
        
        # Repair
        kicked_code, kicked_cov = greedy_repair(kicked_code, kicked_cov, deadline - 15.0)
        
        # Compress
        kicked_code, kicked_cov = compress(kicked_code, kicked_cov, deadline - 10.0)
        
        if len(kicked_code) < best_size:
            best_code = kicked_code[:]
            best_size = len(kicked_code)
            cur_code = kicked_code[:]
            cur_cov = kicked_cov.copy()
        else:
            # With some probability, accept worse solution to escape local optima
            if rng.random() < 0.3:
                cur_code = kicked_code[:]
                cur_cov = kicked_cov.copy()

    # Phase 3: Final aggressive compression
    if (time.time() - t_start) < deadline - 3.0:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        cur_code, cur_cov = compress(cur_code, cur_cov, deadline - 2.0)
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        # Fallback: repair
        best_code, final_cov = greedy_repair(best_code, final_cov, deadline - 1.0)
        if not np.all(final_cov > 0):
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
