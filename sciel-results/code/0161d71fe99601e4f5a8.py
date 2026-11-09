"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute bitsets and adjacency for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        # 1. Build Quad structures
        all_quads = list(combinations(range(target_n), 4))
        quad_bitsets = []
        # For each triple, store the bitsets of the quads it belongs to, 
        # excluding the triple's own bit (to check "others")
        triple_in_quads_masks = [[] for _ in range(num_triples)]
        
        for a, b, c, d in all_quads:
            i1 = triple_to_idx[(a,b,c)]
            i2 = triple_to_idx[(a,b,d)]
            i3 = triple_to_idx[(a,c,d)]
            i4 = triple_to_idx[(b,c,d)]
            
            full_mask = (1 << i1) | (1 << i2) | (1 << i3) | (1 << i4)
            quad_bitsets.append(full_mask)
            
            # For delta calc: if we flip i1, we need to check if i2,i3,i4 are all red.
            # Mask of others = full_mask ^ (1 << i1)
            others_mask = full_mask ^ (1 << i1)
            triple_in_quads_masks[i1].append(others_mask)
            
            others_mask = full_mask ^ (1 << i2)
            triple_in_quads_masks[i2].append(others_mask)
            
            others_mask = full_mask ^ (1 << i3)
            triple_in_quads_masks[i3].append(others_mask)
            
            others_mask = full_mask ^ (1 << i4)
            triple_in_quads_masks[i4].append(others_mask)

        # 2. Build Quint structures
        all_quints = list(combinations(range(target_n), 5))
        quint_bitsets = []
        triple_in_quints_masks = [[] for _ in range(num_triples)]
        
        for q5 in all_quints:
            full_mask = 0
            indices = []
            for t in combinations(q5, 3):
                idx = triple_to_idx[t]
                full_mask |= (1 << idx)
                indices.append(idx)
            
            quint_bitsets.append(full_mask)
            
            for idx in indices:
                others_mask = full_mask ^ (1 << idx)
                triple_in_quints_masks[idx].append(others_mask)

        return (triple_list, num_triples, quad_bitsets, quint_bitsets, 
                triple_in_quads_masks, triple_in_quints_masks)

    def local_search_bitset(target_n, init_red_mask, deadline):
        """
        Run local search using bitsets.
        init_red_mask: Python int where bit i is 1 if triple i is Red (0), 0 if Blue (1).
        Returns: (final_red_mask, success)
        """
        triple_list, num_triples, quad_bitsets, quint_bitsets, \
        triple_in_quads_masks, triple_in_quints_masks = build_structures(target_n)
        
        red_mask = init_red_mask
        num_quads = len(quad_bitsets)
        num_quints = len(quint_bitsets)
        
        # Initial violation counts
        # Red K4: (red_mask & qbs) == qbs
        # Blue K5: (red_mask & qbs) == 0
        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1
                
        if rk4 == 0 and bk5 == 0:
            return red_mask, True

        max_iters = 100000
        temp_start = 2.0
        temp_end = 0.001
        best_violations = rk4 + bk5
        stuck_counter = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return red_mask, True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip
            tidx = -1
            r = random.random()
            
            # Strategy: Sample random quads/quints to find violations
            if rk4 > 0 and (bk5 == 0 or r < 0.5):
                # Try to find a red K4
                for _ in range(20):
                    qi = random.randint(0, num_quads - 1)
                    qbs = quad_bitsets[qi]
                    if (red_mask & qbs) == qbs:
                        # It's a violation. Pick a random triple from it.
                        # We need the indices. We didn't store them per quad in bitset form easily for random pick.
                        # Re-derive indices from bitset? Or store indices.
                        # Better: store quad_indices list too.
                        # For now, let's just pick a random bit from the mask.
                        # To pick a random set bit:
                        # tmp = qbs
                        # while True:
                        #   # random shift? 
                        # Simpler: We stored triple_in_quads, but not quad_in_triples.
                        # Let's just pick a random triple from the whole space if this fails often, 
                        # or precompute quad_indices.
                        # Precomputation is better.
                        pass
                
                # Fallback if we didn't find one in 20 tries (unlikely if rk4>0 but possible)
                # Or just pick random triple
                tidx = random.randint(0, num_triples - 1)
                
            elif bk5 > 0:
                # Try to find a blue K5
                for _ in range(20):
                    qi = random.randint(0, num_quints - 1)
                    qbs = quint_bitsets[qi]
                    if (red_mask & qbs) == 0:
                        # It's a violation.
                        pass
                tidx = random.randint(0, num_triples - 1)
            else:
                tidx = random.randint(0, num_triples - 1)

            # To make the "pick from violating" strategy work efficiently, 
            # we should have stored the indices for each quad/quint.
            # Let's modify build_structures to return indices as well.
            # But for this draft, I will assume the random pick is sufficient 
            # or I will add a small helper.
            
            # Actually, picking a random triple globally is weak.
            # Let's re-do build_structures to include indices for quads/quints.
            
            old_color_bit = (red_mask >> tidx) & 1
            
            # Calculate Delta
            delta_rk4 = 0
            for others_mask in triple_in_quads_masks[tidx]:
                # If old is Red (1), and others are all Red -> removing this bit breaks a K4 (delta -1)
                # If old is Blue (0), and others are all Red -> adding this bit creates a K4 (delta +1)
                if (red_mask & others_mask) == others_mask:
                    if old_color_bit == 1:
                        delta_rk4 -= 1
                    else:
                        delta_rk4 += 1
            
            delta_bk5 = 0
            for others_mask in triple_in_quints_masks[tidx]:
                # If old is Blue (0), and others are all Blue (0 in red_mask) -> removing this bit breaks a K5 (delta -1)
                # If old is Red (1), and others are all Blue (0 in red_mask) -> adding this bit creates a K5 (delta +1)
                # Check if others are all Blue: (red_mask & others_mask) == 0
                if (red_mask & others_mask) == 0:
                    if old_color_bit == 0:
                        delta_bk5 -= 1
                    else:
                        delta_bk5 += 1
                        
            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                # Accept
                red_mask ^= (1 << tidx)
                rk4 += delta_rk4
                bk5 += delta_bk5
                
                if rk4 + bk5 < best_violations:
                    best_violations = rk4 + bk5
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
                
            # Perturb if stuck
            if stuck_counter > 5000:
                stuck_counter = 0
                # Flip ~5% of bits
                for _ in range(num_triples // 20):
                    p = random.randint(0, num_triples - 1)
                    red_mask ^= (1 << p)
                # Recompute
                rk4 = 0
                for qbs in quad_bitsets:
                    if (red_mask & qbs) == qbs:
                        rk4 += 1
                bk5 = 0
                for qbs in quint_bitsets:
                    if (red_mask & qbs) == 0:
                        bk5 += 1

        return red_mask, (rk4 == 0 and bk5 == 0)

    # Modified build_structures to return indices for better sampling
    def build_structures_v2(target_n):
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_bitsets = []
        quad_indices = []
        triple_in_quads_masks = [[] for _ in range(num_triples)]
        
        for a, b, c, d in all_quads:
            i1 = triple_to_idx[(a,b,c)]
            i2 = triple_to_idx[(a,b,d)]
            i3 = triple_to_idx[(a,c,d)]
            i4 = triple_to_idx[(b,c,d)]
            inds = [i1, i2, i3, i4]
            quad_indices.append(inds)
            
            full_mask = 0
            for i in inds:
                full_mask |= (1 << i)
            quad_bitsets.append(full_mask)
            
            for i in inds:
                others_mask = full_mask ^ (1 << i)
                triple_in_quads_masks[i].append(others_mask)

        all_quints = list(combinations(range(target_n), 5))
        quint_bitsets = []
        quint_indices = []
        triple_in_quints_masks = [[] for _ in range(num_triples)]
        
        for q5 in all_quints:
            full_mask = 0
            inds = []
            for t in combinations(q5, 3):
                idx = triple_to_idx[t]
                full_mask |= (1 << idx)
                inds.append(idx)
            
            quint_bitsets.append(full_mask)
            quint_indices.append(inds)
            
            for i in inds:
                others_mask = full_mask ^ (1 << i)
                triple_in_quints_masks[i].append(others_mask)

        return (triple_list, num_triples, quad_bitsets, quad_indices, 
                quint_bitsets, quint_indices, triple_in_quads_masks, triple_in_quints_masks)

    def local_search_bitset_v2(target_n, init_red_mask, deadline):
        triple_list, num_triples, quad_bitsets, quad_indices, \
        quint_bitsets, quint_indices, triple_in_quads_masks, triple_in_quints_masks = build_structures_v2(target_n)
        
        red_mask = init_red_mask
        num_quads = len(quad_bitsets)
        num_quints = len(quint_bitsets)
        
        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1
                
        if rk4 == 0 and bk5 == 0:
            return red_mask, True

        max_iters = 100000
        temp_start = 2.0
        temp_end = 0.001
        best_violations = rk4 + bk5
        stuck_counter = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return red_mask, True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            tidx = -1
            r = random.random()
            
            if rk4 > 0 and (bk5 == 0 or r < 0.5):
                # Sample random quads to find a violation
                for _ in range(10):
                    qi = random.randint(0, num_quads - 1)
                    if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                        tidx = quad_indices[qi][random.randint(0, 3)]
                        break
            elif bk5 > 0:
                # Sample random quints to find a violation
                for _ in range(10):
                    qi = random.randint(0, num_quints - 1)
                    if (red_mask & quint_bitsets[qi]) == 0:
                        tidx = quint_indices[qi][random.randint(0, 9)]
                        break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color_bit = (red_mask >> tidx) & 1
            
            delta_rk4 = 0
            for others_mask in triple_in_quads_masks[tidx]:
                if (red_mask & others_mask) == others_mask:
                    if old_color_bit == 1:
                        delta_rk4 -= 1
                    else:
                        delta_rk4 += 1
            
            delta_bk5 = 0
            for others_mask in triple_in_quints_masks[tidx]:
                if (red_mask & others_mask) == 0:
                    if old_color_bit == 0:
                        delta_bk5 -= 1
                    else:
                        delta_bk5 += 1
                        
            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                red_mask ^= (1 << tidx)
                rk4 += delta_rk4
                bk5 += delta_bk5
                
                if rk4 + bk5 < best_violations:
                    best_violations = rk4 + bk5
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
                
            if stuck_counter > 5000:
                stuck_counter = 0
                for _ in range(num_triples // 20):
                    p = random.randint(0, num_triples - 1)
                    red_mask ^= (1 << p)
                rk4 = 0
                for qbs in quad_bitsets:
                    if (red_mask & qbs) == qbs:
                        rk4 += 1
                bk5 = 0
                for qbs in quint_bitsets:
                    if (red_mask & qbs) == 0:
                        bk5 += 1

        return red_mask, (rk4 == 0 and bk5 == 0)

    def colors_from_mask(n, red_mask, triple_list):
        colors = []
        for i, t in enumerate(triple_list):
            colors.append(0 if (red_mask >> i) & 1 else 1)
        return colors

    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        # Precompute structures once per target_n
        structures = build_structures_v2(target_n)
        triple_list, num_triples = structures[0], structures[1]
        
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Diverse Initializations
            init_red_mask = 0
            if restart == 0:
                # Mod 3
                for i, (a,b,c) in enumerate(triple_list):
                    if (a + b + c) % 3 == 0:
                        init_red_mask |= (1 << i)
            elif restart == 1:
                # Mod 4
                for i, (a,b,c) in enumerate(triple_list):
                    if (a + b + c) % 4 == 0:
                        init_red_mask |= (1 << i)
            elif restart == 2:
                # Random
                for i in range(num_triples):
                    if random.random() < 0.5:
                        init_red_mask |= (1 << i)
            elif restart == 3:
                # Bias Red
                for i in range(num_triples):
                    if random.random() < 0.4:
                        init_red_mask |= (1 << i)
            elif restart == 4:
                # Bias Blue
                for i in range(num_triples):
                    if random.random() < 0.6:
                        init_red_mask |= (1 << i)
            else:
                # Perturb Mod 3
                for i, (a,b,c) in enumerate(triple_list):
                    if (a + b + c) % 3 == 0:
                        if random.random() > 0.1:
                            init_red_mask |= (1 << i)
                    else:
                        if random.random() < 0.1:
                            init_red_mask |= (1 << i)

            final_mask, success = local_search_bitset_v2(target_n, init_red_mask, deadline)
            
            if success:
                found = True
                best_colors = colors_from_mask(target_n, final_mask, triple_list)
                # Verify
                if is_valid(target_n, best_colors):
                    best_n = target_n
                    break
                else:
                    # Should not happen if bitset logic is correct, but safety
                    pass

        if found:
            continue # Move to next n

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
