import time
import random

def construct():
    n = 256
    deadline = time.time() + 140  # Safety margin
    
    # Precompute all 4-APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(ap_list)
    
    # Precompute membership: for each index, which APs contain it
    membership = [[] for _ in range(n)]
    for k, ap in enumerate(ap_list):
        for idx in ap:
            membership[idx].append(k)
            
    def count_mono(colors):
        count = 0
        for ap in ap_list:
            c0 = colors[ap[0]]
            if c0 == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
        
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    best_colors = None
    best_score = float('inf')
    
    # Force first run to be Thue-Morse to ensure a strong baseline
    first_run = True
    
    while time.time() < deadline:
        if first_run:
            colors = thue_morse(n)
            first_run = False
        else:
            # Mix of random and perturbed best
            if random.random() < 0.5 and best_colors is not None:
                # Perturb best
                colors = best_colors[:]
                for _ in range(10):
                    i = random.randint(0, n - 1)
                    colors[i] = 1 - colors[i]
            else:
                # Random
                colors = [random.randint(0, 1) for _ in range(n)]
            
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
            
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx = -1
            
            for i in range(n):
                delta = 0
                ci = colors[i]
                for k in membership[i]:
                    ap = ap_list[k]
                    a, b, c, d = ap
                    if i == a:
                        o1, o2, o3 = b, c, d
                    elif i == b:
                        o1, o2, o3 = a, c, d
                    elif i == c:
                        o1, o2, o3 = a, b, d
                    else:
                        o1, o2, o3 = a, b, c
                    
                    co1 = colors[o1]
                    if co1 == colors[o2] == colors[o3]:
                        if co1 == ci:
                            delta -= 1
                        else:
                            delta += 1
                            
                if delta < best_delta:
                    best_delta = delta
                    best_idx = i
                    
            if best_idx != -1:
                colors[best_idx] = 1 - colors[best_idx]
                current_score += best_delta
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    
    return best_colors

# EVOLVE-BLOCK-END
