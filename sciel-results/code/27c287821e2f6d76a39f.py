import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    # We hypothesize 3 colors are sufficient.
    # If 3 colors fail, we might need 4, but 3 is the theoretical minimum for n=128.
    # Let's try to find a 3-coloring.
    
    # Precompute 4-APs
    ap_list = []
    pos_to_aps = [[] for _ in range(n)]
    
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            idx2 = a + d
            idx3 = a + 2*d
            idx4 = a + 3*d
            if idx4 < n:
                ap_idx = len(ap_list)
                ap_list.append((a, idx2, idx3, idx4))
                pos_to_aps[a].append(ap_idx)
                pos_to_aps[idx2].append(ap_idx)
                pos_to_aps[idx3].append(ap_idx)
                pos_to_aps[idx4].append(ap_idx)
                
    def is_bad(c1, c2, c3, c4):
        if c1 == c2:
            if c1 == c3:
                return True
            if c3 == c4:
                return True
            return c4 == c1
        else:
            if c1 == c3:
                return c2 == c4
            else:
                if c2 == c3:
                    return c4 == c1
                return False

    def get_energy(colors):
        e = 0
        for ap in ap_list:
            if is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                e += 1
        return e

    best_colors = None
    best_energy = float('inf')
    
    # Multiple restarts to escape local minima
    for restart in range(20):
        colors = [random.randint(0, 2) for _ in range(n)]
        energy = get_energy(colors)
        
        if energy < best_energy:
            best_energy = energy
            best_colors = colors.copy()
            if best_energy == 0:
                return best_colors

        T = 1.0
        cooling = 0.999
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            i = random.randint(0, n - 1)
            current_color = colors[i]
            
            best_delta = float('inf')
            best_new_color = -1
            
            for new_color in [0, 1, 2]:
                if new_color == current_color:
                    continue
                
                delta = 0
                for ap_idx in pos_to_aps[i]:
                    ap = ap_list[ap_idx]
                    c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    
                    # Determine position of i in the AP to swap correctly
                    # This is inefficient. Better to just recalculate the badness of the AP.
                    # But we need to know if it WAS bad and IF it IS bad.
                    
                    was_bad = is_bad(c1, c2, c3, c4)
                    
                    # Apply change
                    # Find index of i in ap
                    # This is also slow.
                    # Let's store the AP as a tuple and just re-evaluate.
                    # To speed up, we can precompute the position of each index in each AP it belongs to?
                    # Or just check the 4 values.
                    
                    # Let's optimize by storing the AP indices and checking which one is i.
                    # Since pos_to_aps[i] gives the AP indices, we can just fetch the AP.
                    # We need to update the color in a temporary view.
                    
                    # Actually, the delta calculation above using `colors[i] = new_color` and restoring is fine.
                    # Let's stick to that but use the fast is_bad.
                    
                    old_c = colors[i]
                    colors[i] = new_color
                    is_bad_now = is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                    colors[i] = old_c
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_delta <= 0:
                colors[i] = best_new_color
                energy += best_delta
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return best_colors
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                if random.random() < math.exp(-best_delta / T):
                    colors[i] = best_new_color
                    energy += best_delta
                no_improve_count += 1
            
            T *= cooling
            if T < 0.05:
                T = 0.05
            
            if no_improve_count > max_no_improve:
                break

    return best_colors
