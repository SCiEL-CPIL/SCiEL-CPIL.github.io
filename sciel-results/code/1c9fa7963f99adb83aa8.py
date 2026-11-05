import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safe margin before 150s kill

    def get_triple_indices(n):
        """Return list of all triples (a,b,c) with a<b<c."""
        return list(combinations(range(n), 3))

    def build_k4_k5(n):
        """Build K4 and K5 structures for fast violation checking."""
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # K4: 4 vertices -> 4 triples
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        # K5: 5 vertices -> 10 triples
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Reverse indices: triple_idx -> list of clique indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations_bitmask(colors, k4s, k5s):
        """Count red K4s and blue K5s using bitwise operations."""
        # colors is a bitmask where bit i is 1 if color is 1 (blue), 0 if red.
        # We need to check if a set of bits are all 0 (Red K4) or all 1 (Blue K5).
        
        red_count = 0
        for idxs in k4s:
            # If all 4 bits are 0, then (colors & mask) == 0
            # Construct mask for these 4 bits
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            if (colors & mask) == 0:
                red_count += 1
        
        blue_count = 0
        for idxs in k5s:
            # If all 10 bits are 1, then (colors & mask) == mask
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            if (colors & mask) == mask:
                blue_count += 1
                
        return red_count + blue_count

    def delta_cost_bitmask(colors, t, k4s, k5s, triple_to_k4s, triple_to_k5s):
        """Compute cost delta when flipping triple t."""
        delta = 0
        bit = 1 << t
        is_blue = (colors & bit) != 0

        # Check K4s containing t
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            
            # Count how many are 0 (Red) in the current state
            # If we flip t, we change its contribution.
            # Current red count in this K4:
            # It's faster to just check the condition before and after?
            # Let's count reds in the K4 excluding t, then add t's status.
            others_mask = mask & ~bit
            reds_others = np.count_nonzero(np.array([(colors >> i) & 1 == 0 for i in idxs if i != t]))
            
            if is_blue:
                # t was Blue (1). Flipping to Red (0).
                # Before: reds_others. After: reds_others + 1.
                if reds_others == 3:
                    # Becomes 4 reds -> Violation created
                    delta += 1
                elif reds_others == 4:
                    # Impossible, max 3 others
                    pass
                else:
                    # No violation created or removed?
                    # If reds_others < 3, no violation.
                    pass
            else:
                # t was Red (0). Flipping to Blue (1).
                # Before: reds_others + 1. After: reds_others.
                if reds_others == 3:
                    # Was 4 reds (Violation), becomes 3. Violation removed.
                    delta -= 1
                else:
                    pass

        # Check K5s containing t
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            
            # Count blues in K4 excluding t
            blues_others = np.count_nonzero(np.array([(colors >> i) & 1 == 1 for i in idxs if i != t]))
            
            if is_blue:
                # t was Blue. Flipping to Red.
                # Before: blues_others + 1. After: blues_others.
                if blues_others == 9:
                    # Was 10 blues (Violation), becomes 9. Violation removed.
                    delta -= 1
            else:
                # t was Red. Flipping to Blue.
                # Before: blues_others. After: blues_others + 1.
                if blues_others == 9:
                    # Becomes 10 blues -> Violation created
                    delta += 1

        return delta

    def structured_init_bitmask(n, variant=0, seed=0):
        """Generate structured initial colorings as bitmask."""
        triples = get_triple_indices(n)
        num_triples = len(triples)
        
        # We use a Python integer as a bitmask.
        # Bit i corresponds to triple i. 1 = Blue, 0 = Red.
        
        if variant == 0:
            # (a+b+c) mod 3
            mask = 0
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 3 != 0: # Blue if not 0 mod 3? Let's say 1 is Blue.
                    mask |= (1 << i)
            return mask
        elif variant == 1:
            # (a+b+2*c) mod 3
            mask = 0
            for i, (a, b, c) in enumerate(triples):
                if (a + b + 2*c) % 3 != 0:
                    mask |= (1 << i)
            return mask
        elif variant == 2:
            # (a*b + b*c + c*a) mod 2
            mask = 0
            for i, (a, b, c) in enumerate(triples):
                if (a*b + b*c + c*a) % 2 == 1:
                    mask |= (1 << i)
            return mask
        elif variant == 3:
            # (a+b+c) mod 2
            mask = 0
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 2 == 1:
                    mask |= (1 << i)
            return mask
        elif variant == 4:
            # Random
            np.random.seed(seed)
            mask = 0
            for i in range(num_triples):
                if np.random.rand() < 0.5:
                    mask |= (1 << i)
            return mask
        else:
            # Default
            mask = 0
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 3 != 0:
                    mask |= (1 << i)
            return mask

    def local_search_bitmask(n, time_limit, seed=0, init_variant=0, k4s=None, k5s=None, triple_to_k4s=None, triple_to_k5s=None):
        """Tabu Search with LNS."""
        if k4s is None:
            _, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_k4_k5(n)
        else:
            num_triples = len(get_triple_indices(n))
            
        np.random.seed(seed)
        colors = structured_init_bitmask(n, init_variant, seed)
        
        # Precompute masks for K4s and K5s for faster counting if needed, 
        # but delta_cost does it on the fly. 
        # For efficiency in delta, we might want to precompute "others" masks? 
        # No, the loop in delta is small (4 or 10 elements).
        
        current_cost = count_violations_bitmask(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors

        tabu_tenure = 10
        tabu_list = {} # triple_idx -> time when it becomes non-tabu
        
        step = 0
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < time_limit:
            step += 1
            
            # Find a move
            # 1. Try to find a non-tabu move that improves cost
            # 2. If none, pick best non-tabu move
            # 3. If all tabu, pick best move (aspiration)
            
            best_move_t = -1
            best_move_delta = 0
            best_move_cost = current_cost
            
            # Optimization: Instead of checking all triples, check a random subset or those involved in violations
            # For now, check all to ensure correctness, but this is O(N).
            # N is ~ 5000 for n=30. 5000 * 150s is fine.
            
            # To speed up, we can prioritize triples in violated cliques.
            # Let's just iterate all triples for simplicity in this structural change, 
            # but use a random order to avoid bias.
            
            indices = np.random.permutation(num_triples)
            
            for t in indices:
                if t in tabu_list and tabu_list[t] > step:
                    continue
                
                old_color_bit = (colors >> t) & 1
                dc = delta_cost_bitmask(colors, t, k4s, k5s, triple_to_k4s, triple_to_k5s)
                new_cost = current_cost + dc
                
                # Aspiration: accept if it improves best cost even if tabu
                if t in tabu_list and tabu_list[t] > step:
                    if new_cost < best_cost:
                        pass # Allow
                    else:
                        continue
                        
                if dc < best_move_delta or (dc == best_move_delta and best_move_t == -1):
                    best_move_delta = dc
                    best_move_t = t
                    best_move_cost = new_cost
                    
                # Early exit if we found a negative delta (improvement)
                if dc < 0:
                    break

            if best_move_t == -1:
                # No valid move found? Should not happen unless all are tabu and none improve best.
                # If we get here, we are stuck.
                # Reset tabu or do a random jump.
                break
                
            # Perform move
            colors ^= (1 << best_move_t)
            current_cost = best_move_cost
            
            # Update Tabu List
            tabu_list[best_move_t] = step + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors
                no_improve = 0
                if best_cost == 0:
                    return best_colors, True
            else:
                no_improve += 1
                
            # LNS Perturbation
            if no_improve > max_no_improve:
                # Unfix a random set of triples and re-optimize? 
                # Or just random restart?
                # Let's do a "shake": flip 10% of random triples.
                num_shake = max(1, num_triples // 10)
                shake_indices = np.random.choice(num_triples, size=num_shake, replace=False)
                for t in shake_indices:
                    colors ^= (1 << t)
                    # Update cost roughly or just recalculate?
                    # Recalculating is O(N). Let's just recalculate current_cost.
                
                current_cost = count_violations_bitmask(colors, k4s, k5s)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors
                else:
                    # Revert to best if shake was bad? Or keep going?
                    # Tabu search usually keeps going. But if it's much worse, revert.
                    if current_cost > best_cost * 2:
                        colors = best_colors
                        current_cost = best_cost
                        
                no_improve = 0
                tabu_list.clear()
                
            if no_improve > max_no_improve * 5:
                # Full restart with new seed
                break

        return best_colors, (best_cost == 0)

    def verify(n, colors_bitmask):
        """Verify coloring has no red K4 and no blue K5."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            i1 = triple_to_idx[(a, b, c)]
            i2 = triple_to_idx[(a, b, d)]
            i3 = triple_to_idx[(a, c, d)]
            i4 = triple_to_idx[(b, c, d)]
            
            # Check if all are Red (0)
            if ((colors_bitmask >> i1) & 1) == 0 and \
               ((colors_bitmask >> i2) & 1) == 0 and \
               ((colors_bitmask >> i3) & 1) == 0 and \
               ((colors_bitmask >> i4) & 1) == 0:
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                idx = triple_to_idx[triple]
                if ((colors_bitmask >> idx) & 1) != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    target_ns = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 3:
            break

        # Allocate time
        if n == 35:
            time_budget = min(60.0, remaining * 0.7)
        elif n == 34:
            time_budget = min(40.0, remaining * 0.6)
        elif n >= 30:
            time_budget = min(20.0, remaining * 0.5)
        elif n >= 25:
            time_budget = min(15.0, remaining * 0.4)
        elif n >= 20:
            time_budget = min(10.0, remaining * 0.3)
        else:
            time_budget = min(5.0, remaining * 0.2)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False
        num_seeds = 20 if n >= 35 else 15
        
        # Pre-build structures for this n
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_k4_k5(n)

        for seed in range(num_seeds):
            if time.time() > time_limit:
                break

            variant = seed % 5  # Cycle through 5 structured inits

            try:
                colors_bitmask, is_valid = local_search_bitmask(n, time_limit, seed=seed, init_variant=variant, 
                                                                k4s=k4s, k5s=k5s, triple_to_k4s=triple_to_k4s, triple_to_k5s=triple_to_k5s)

                if is_valid:
                    if verify(n, colors_bitmask):
                        # Convert bitmask to list of colors
                        colors_list = [int((colors_bitmask >> i) & 1) for i in range(num_triples)]
                        final_n = n
                        final_colors = colors_list
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    if final_n == 0 or not final_colors:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

        if verify(n, 0): # Verify needs bitmask. Let's convert.
            pass # This is a fallback, just return something valid.
        
        # Fallback: Just return a small valid coloring
        # For n=9, (a+b+c)%3 coloring works?
        # Let's construct bitmask for n=9
        mask = 0
        for i, (a, b, c) in enumerate(triples):
             if (a + b + c) % 3 != 0:
                 mask |= (1 << i)
        if verify(9, mask):
            final_n = 9
            final_colors = [int((mask >> i) & 1) for i in range(len(triples))]
        else:
            final_n = 1
            final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
