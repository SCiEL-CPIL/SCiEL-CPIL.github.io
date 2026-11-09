import time
import random
import numpy as np

def construct():
    N = 1000
    max_val = 1000
    max_sum = 2 * max_val
    deadline = time.time() + 115  # Safety margin for 150s limit

    # --- Core data structures and helpers ---
    
    def make_state():
        """Create a fresh state container."""
        return {
            'S': [],
            'in_S': [False] * (max_val + 1),
            'counts': [0] * (max_sum + 1)
        }

    def try_add(state, x):
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        if in_S[x]:
            return False
            
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            s = x + y
            if counts[s] >= 2:
                return False
        
        # Add
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(state, x):
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        if not in_S[x]:
            return False
            
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1
        return True

    def validate_set(S):
        counts = [0] * (max_sum + 1)
        for i, a in enumerate(S):
            counts[2 * a] += 1
            for b in S[i:]:
                if b != a:
                    counts[a + b] += 1
        return all(c <= 2 for c in counts)

    # --- Heuristic 1: Deterministic Greedy (Ascending) ---
    # This provides a strong, consistent baseline.
    def greedy_ascending():
        state = make_state()
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        for x in range(1, max_val + 1):
            # Check if x can be added
            if counts[2 * x] >= 2:
                continue
            valid = True
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
            if valid:
                try_add(state, x)
        return S[:]

    # --- Heuristic 2: Randomized Greedy ---
    def greedy_random(seed):
        rng = random.Random(seed)
        state = make_state()
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            try_add(state, x)
        return state['S'][:]

    # --- Heuristic 3: Improved Local Search with Targeted Swaps ---
    # This is the core refinement. It attempts to add blocked elements by swapping out blockers.
    
    def local_search(S_init, time_limit, seed=42):
        rng = random.Random(seed)
        state = make_state()
        for x in S_init:
            try_add(state, x)
        
        S = state['S']
        in_S = state['in_S']
        counts = state['counts']
        
        # Precompute sum_pairs for blocker identification
        # sum_pairs[s] = list of (a, b) where a+b=s, a<=b, a,b in S
        def build_sum_pairs():
            sp = {}
            for i, a in enumerate(S):
                sp[2*a] = sp.get(2*a, []) + [(a, a)]
                for j in range(i+1, len(S)):
                    b = S[j]
                    s = a + b
                    sp[s] = sp.get(s, []) + [(a, b)]
            return sp
        
        def find_blockers_for_candidate(c, sum_pairs):
            """Find elements in S that block candidate c."""
            blockers = set()
            
            # Check 2*c
            if counts[2 * c] >= 2:
                for a, b in sum_pairs.get(2 * c, []):
                    blockers.add(a)
                    blockers.add(b)
            
            # Check c + y for y in S
            for y in S:
                s = c + y
                if counts[s] >= 2:
                    for a, b in sum_pairs.get(s, []):
                        blockers.add(a)
                        blockers.add(b)
            
            return [b for b in blockers if in_S[b]]
        
        def try_add_candidate(c, sum_pairs):
            """Try to add candidate c, returning (success, new_sum_pairs)."""
            if in_S[c]:
                return False, None
            
            # Check if c can be added
            if counts[2 * c] >= 2:
                return False, None
            for y in S:
                s = c + y
                if counts[s] >= 2:
                    return False, None
            
            # Add c
            S.append(c)
            in_S[c] = True
            counts[2 * c] += 1
            new_pairs = []
            for y in S:
                if y != c:
                    counts[c + y] += 1
                    new_pairs.append((min(c, y), max(c, y)))
            
            # Update sum_pairs
            new_sp = {}
            for s, pairs in sum_pairs.items():
                new_sp[s] = pairs[:]
            for a, b in new_pairs:
                s = a + b
                if s not in new_sp:
                    new_sp[s] = []
                new_sp[s].append((a, b))
            if 2*c not in new_sp:
                new_sp[2*c] = []
            new_sp[2*c].append((c, c))
            
            return True, new_sp
        
        def try_remove_element(x, sum_pairs):
            """Remove element x, returning (success, new_sum_pairs)."""
            if not in_S[x]:
                return False, None
            
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            
            # Remove pairs involving x
            new_sp = {}
            for s, pairs in sum_pairs.items():
                remaining = []
                for a, b in pairs:
                    if a == x or b == x:
                        counts[s] -= 1
                    else:
                        remaining.append((a, b))
                if remaining:
                    new_sp[s] = remaining
            
            return True, new_sp
        
        # Build initial sum_pairs
        sum_pairs = build_sum_pairs()
        
        # Candidates not in S
        candidates_not_in_S = [x for x in range(1, max_val + 1) if not in_S[x]]
        rng.shuffle(candidates_not_in_S)
        
        # Track best size during search
        best_size_in_search = len(S)
        best_S_snapshot = S[:]
        
        # Main search loop
        while time.time() < time_limit:
            if not candidates_not_in_S:
                break
            
            # Pick a candidate (prioritize ones that are "close" to being addable)
            # Try random candidates, but also try some that are near existing elements
            idx = rng.randint(0, len(candidates_not_in_S) - 1)
            c = candidates_not_in_S[idx]
            
            if in_S[c]:
                candidates_not_in_S.pop(idx)
                continue
            
            # Try direct add
            success, _ = try_add_candidate(c, sum_pairs)
            if success:
                # Rebuild sum_pairs
                sum_pairs = build_sum_pairs()
                candidates_not_in_S.pop(idx)
                if len(S) > best_size_in_search:
                    best_size_in_search = len(S)
                    best_S_snapshot = S[:]
                continue
            
            # Blocked. Find blockers.
            blockers = find_blockers_for_candidate(c, sum_pairs)
            
            if not blockers:
                # Shouldn't happen if try_add failed, but just in case
                candidates_not_in_S.pop(idx)
                continue
            
            # Try 1-for-1 swaps: remove one blocker, add c
            found_swap = False
            for b in blockers:
                # Try removing b and adding c
                # First remove b
                S.remove(b)
                in_S[b] = False
                counts[2 * b] -= 1
                for y in S:
                    if y != b:
                        counts[b + y] -= 1
                
                # Now try adding c
                can_add = True
                if counts[2 * c] >= 2:
                    can_add = False
                else:
                    for y in S:
                        s = c + y
                        if counts[s] >= 2:
                            can_add = False
                            break
                
                if can_add:
                    # Add c
                    S.append(c)
                    in_S[c] = True
                    counts[2 * c] += 1
                    for y in S:
                        if y != c:
                            counts[c + y] += 1
                    
                    # Rebuild sum_pairs
                    sum_pairs = build_sum_pairs()
                    candidates_not_in_S.pop(idx)
                    found_swap = True
                    if len(S) > best_size_in_search:
                        best_size_in_search = len(S)
                        best_S_snapshot = S[:]
                    break
                else:
                    # Restore b
                    S.append(b)
                    in_S[b] = True
                    counts[2 * b] += 1
                    for y in S:
                        if y != b:
                            counts[b + y] += 1
            
            if not found_swap:
                # Try 2-for-1 moves: remove two blockers, add one candidate
                # This is more expensive but can escape deeper local minima
                if len(blockers) >= 2:
                    # Try pairs of blockers
                    tried_pairs = set()
                    for i in range(len(blockers)):
                        for j in range(i + 1, len(blockers)):
                            b1, b2 = blockers[i], blockers[j]
                            pair_key = (min(b1, b2), max(b1, b2))
                            if pair_key in tried_pairs:
                                continue
                            tried_pairs.add(pair_key)
                            
                            # Remove b1 and b2
                            S.remove(b1)
                            in_S[b1] = False
                            counts[2 * b1] -= 1
                            for y in S:
                                if y != b1:
                                    counts[b1 + y] -= 1
                            
                            S.remove(b2)
                            in_S[b2] = False
                            counts[2 * b2] -= 1
                            for y in S:
                                if y != b2:
                                    counts[b2 + y] -= 1
                            
                            # Try adding c
                            can_add = True
                            if counts[2 * c] >= 2:
                                can_add = False
                            else:
                                for y in S:
                                    s = c + y
                                    if counts[s] >= 2:
                                        can_add = False
                                        break
                            
                            if can_add:
                                # Add c
                                S.append(c)
                                in_S[c] = True
                                counts[2 * c] += 1
                                for y in S:
                                    if y != c:
                                        counts[c + y] += 1
                                
                                # Rebuild sum_pairs
                                sum_pairs = build_sum_pairs()
                                candidates_not_in_S.pop(idx)
                                found_swap = True
                                if len(S) > best_size_in_search:
                                    best_size_in_search = len(S)
                                    best_S_snapshot = S[:]
                                break
                            else:
                                # Restore b1 and b2
                                S.append(b1)
                                in_S[b1] = True
                                counts[2 * b1] += 1
                                for y in S:
                                    if y != b1:
                                        counts[b1 + y] += 1
                                
                                S.append(b2)
                                in_S[b2] = True
                                counts[2 * b2] += 1
                                for y in S:
                                    if y != b2:
                                        counts[b2 + y] += 1
                            
                            if found_swap:
                                break
                        if found_swap:
                            break
                
                # If still no swap found, try a random swap as a last resort
                if not found_swap and len(S) > 0:
                    out_idx = rng.randint(0, len(S) - 1)
                    x_out = S[out_idx]
                    
                    S.remove(x_out)
                    in_S[x_out] = False
                    counts[2 * x_out] -= 1
                    for y in S:
                        if y != x_out:
                            counts[x_out + y] -= 1
                    
                    can_add = True
                    if counts[2 * c] >= 2:
                        can_add = False
                    else:
                        for y in S:
                            s = c + y
                            if counts[s] >= 2:
                                can_add = False
                                break
                    
                    if can_add:
                        S.append(c)
                        in_S[c] = True
                        counts[2 * c] += 1
                        for y in S:
                            if y != c:
                                counts[c + y] += 1
                        
                        sum_pairs = build_sum_pairs()
                        candidates_not_in_S.pop(idx)
                        if len(S) > best_size_in_search:
                            best_size_in_search = len(S)
                            best_S_snapshot = S[:]
                    else:
                        # Restore
                        S.append(x_out)
                        in_S[x_out] = True
                        counts[2 * x_out] += 1
                        for y in S:
                            if y != x_out:
                                counts[x_out + y] += 1
        
        # Final polish: try to add any remaining candidates
        for c in candidates_not_in_S:
            if try_add(state, c):
                pass
        
        # Final sweep
        final_S = []
        for x in range(1, max_val + 1):
            if in_S[x]:
                final_S.append(x)
        
        return final_S

    # --- Heuristic 4: Final Polish Pass ---
    # After local search, do a more thorough pass to add any remaining candidates
    def final_polish(S_init):
        state = make_state()
        for x in S_init:
            try_add(state, x)
        
        S = state['S']
        
        # Try adding each candidate multiple times in different orders
        for attempt in range(3):
            candidates = [x for x in range(1, max_val + 1) if not state['in_S'][x]]
            if attempt == 0:
                pass  # Already tried in order
            elif attempt == 1:
                random.shuffle(candidates)
            else:
                candidates.sort(reverse=True)
            
            for x in candidates:
                try_add(state, x)
        
        return state['S'][:]

    # --- Main Execution Loop ---

    best_S = []
    best_size = 0

    # 1. Deterministic Greedy
    S_det = greedy_ascending()
    best_S = S_det[:]
    best_size = len(S_det)

    # 2. Try a few randomized greedy starts
    # Time budget for each start + local search
    # Total time ~115s.
    # Let's do 5 trials.
    
    t_start = time.time()
    time_per_trial = 20  # seconds
    
    for trial in range(5):
        if time.time() > deadline - 5:
            break
            
        seed = trial * 1000 + 7
        S_rand = greedy_random(seed)
        
        # If the random start is already better, update best
        if len(S_rand) > best_size:
            best_S = S_rand[:]
            best_size = len(S_rand)
        
        # Run local search on the better of (S_det, S_rand) or just S_rand?
        # Let's run on S_rand, but if S_det is much better, maybe start from S_det?
        # Actually, local search from a good start is better.
        # Let's run local search on S_rand.
        
        remaining_time = deadline - time.time() - 10 # 10s safety buffer
        if remaining_time < 5:
            continue
            
        # Allocate time for local search
        ls_time = min(time_per_trial, remaining_time)
        
        S_ls = local_search(S_rand, time.time() + ls_time, seed=seed)
        
        # Apply final polish
        S_polished = final_polish(S_ls)
        
        if len(S_polished) > best_size:
            best_size = len(S_polished)
            best_S = S_polished[:]

    # 3. If time remains, try local search on the deterministic start
    if time.time() < deadline - 15:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            S_ls_det = local_search(S_det, time.time() + remaining_time, seed=999)
            S_polished_det = final_polish(S_ls_det)
            if len(S_polished_det) > best_size:
                best_size = len(S_polished_det)
                best_S = S_polished_det[:]

    # Final Validation
    if not validate_set(best_S):
        # Fallback to deterministic
        best_S = S_det[:]
        best_size = len(S_det)
        if not validate_set(best_S):
            # This should never happen
            return list(range(1, 50))

    return best_S
