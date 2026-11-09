"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill

    n = 256
    # Generate all 4-term arithmetic progressions
    # (a, a+d, a+2d, a+3d)
    aps = []
    ap_indices_by_pos = [[] for _ in range(n)]
    
    # 1-indexed logic for generation, 0-indexed for storage
    for d in range(1, 85):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            ap = (i0, i1, i2, i3)
            ap_idx = len(aps)
            aps.append(ap)
            
            ap_indices_by_pos[i0].append(ap_idx)
            ap_indices_by_pos[i1].append(ap_idx)
            ap_indices_by_pos[i2].append(ap_idx)
            ap_indices_by_pos[i3].append(ap_idx)
            
    total_aps = len(aps)
    
    def count_mono(colors):
        count = 0
        for ap in aps:
            i0, i1, i2, i3 = ap
            c = colors[i0]
            if colors[i1] == c and colors[i2] == c and colors[i3] == c:
                count += 1
        return count

    def get_delta(k, colors):
        # k is 0-indexed
        # Returns change in mono count if we flip colors[k]
        v = colors[k]
        v_new = 1 - v
        
        delta = 0
        for ap_idx in ap_indices_by_pos[k]:
            ap = aps[ap_idx]
            # Get the other 3 indices
            # We know k is one of them
            if ap[0] == k:
                others = (ap[1], ap[2], ap[3])
            elif ap[1] == k:
                others = (ap[0], ap[2], ap[3])
            elif ap[2] == k:
                others = (ap[0], ap[1], ap[3])
            else: # ap[3] == k
                others = (ap[0], ap[1], ap[2])
                
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            # Check current state
            # Is it currently mono?
            curr_mono = (c1 == v and c2 == v and c3 == v)
            
            # Check new state
            # It becomes mono iff others == v_new
            new_mono = (c1 == v_new and c2 == v_new and c3 == v_new)
            
            if curr_mono and not new_mono:
                delta -= 1
            elif not curr_mono and new_mono:
                delta += 1
            # else 0
            
        return delta

    def thue_morse(n):
        """Generate Thue-Morse sequence of length n (0-indexed)."""
        result = []
        for i in range(n):
            # Count of 1-bits in binary representation of i
            result.append(bin(i).count('1') % 2)
        return result

    # Use Thue-Morse as initial coloring (known to have low mono-AP density)
    best_colors = thue_morse(n)
    best_score = count_mono(best_colors)
    
    current_colors = best_colors[:]
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.9998
    min_T = 0.005
    
    # Run SA loop with multiple restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Start with Thue-Morse
            current_colors = thue_morse(n)
            current_score = count_mono(current_colors)
        else:
            # For subsequent restarts, use random perturbation of best
            current_colors = best_colors[:]
            # Perturb 10-20 random positions
            num_perturb = random.randint(10, 20)
            for _ in range(num_perturb):
                k = random.randint(0, n - 1)
                current_colors[k] = 1 - current_colors[k]
            current_score = count_mono(current_colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = current_colors[:]
        
        T = 1.0
        stuck_counter = 0
        no_improve_counter = 0
        
        while time.time() < deadline:
            # Pick random index
            k = random.randint(0, n - 1)
            delta = get_delta(k, current_colors)
            
            # Acceptance criterion
            accept = False
            if delta < 0:
                accept = True
                stuck_counter = 0
                no_improve_counter = 0
            else:
                if delta == 0:
                    accept = True
                    stuck_counter = 0
                else:
                    # Metropolis-Hastings
                    if T > 0.01:
                        prob = math.exp(-delta / T)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        accept = True
                        stuck_counter = 0
            
            if accept:
                current_colors[k] = 1 - current_colors[k]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = current_colors[:]
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                stuck_counter += 1
                no_improve_counter += 1
            
            # Cooling
            T *= cooling
            if T < min_T:
                # Restart with higher T to escape local minima
                T = 1.0
                stuck_counter = 0
                # Perturb current state slightly to jump basin
                for _ in range(3):
                    k2 = random.randint(0, n - 1)
                    d2 = get_delta(k2, current_colors)
                    if d2 <= 0 or random.random() < 0.1:
                        current_colors[k2] = 1 - current_colors[k2]
                        current_score += d2
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = current_colors[:]
                            no_improve_counter = 0

            # If stuck for too long, break to next restart
            if stuck_counter > 3000 or no_improve_counter > 10000:
                break

    return best_colors
# EVOLVE-BLOCK-END
