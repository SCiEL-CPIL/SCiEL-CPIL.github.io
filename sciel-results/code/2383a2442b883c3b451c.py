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

    def run_bitwise_search(target_n, deadline, initial_colors=None):
        """
        Bit-parallel local search for Max-SAT formulation.
        Returns (colors, num_violations)
        """
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Precompute masks
        red_k4_masks = [0] * num_triples
        blue_k5_masks = [0] * num_triples
        red_k4_rev = [[] for _ in range(num_triples)]
        blue_k5_rev = [[] for _ in range(num_triples)]
        
        # Build Red K4 masks
        for a, b, c, d in combinations(range(target_n), 4):
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            
            red_k4_masks[t1] |= (1 << t2) | (1 << t3) | (1 << t4)
            red_k4_masks[t2] |= (1 << t1) | (1 << t3) | (1 << t4)
            red_k4_masks[t3] |= (1 << t1) | (1 << t2) | (1 << t4)
            red_k4_masks[t4] |= (1 << t1) | (1 << t2) | (1 << t3)
            
            red_k4_rev[t2].append(t1)
            red_k4_rev[t3].append(t1)
            red_k4_rev[t4].append(t1)
            
            red_k4_rev[t1].append(t2)
            red_k4_rev[t3].append(t2)
            red_k4_rev[t4].append(t2)
            
            red_k4_rev[t1].append(t3)
            red_k4_rev[t2].append(t3)
            red_k4_rev[t4].append(t3)
            
            red_k4_rev[t1].append(t4)
            red_k4_rev[t2].append(t4)
            red_k4_rev[t3].append(t4)

        # Build Blue K5 masks
        for quint in combinations(range(target_n), 5):
            quads_in_quint = list(combinations(quint, 3))
            triple_indices_in_quint = [triple_to_idx[t] for t in quads_in_quint]
            
            for i, t_idx in enumerate(triple_indices_in_quint):
                mask = 0
                for j in range(10):
                    if j != i:
                        mask |= (1 << triple_indices_in_quint[j])
                        blue_k5_rev[triple_indices_in_quint[j]].append(t_idx)
                blue_k5_masks[t_idx] |= mask

        # Initial Coloring
        # 1 = Red, 0 = Blue in bitset representation?
        # Let's use 1 for Red, 0 for Blue.
        # Task: 0 (red) or 1 (blue).
        # So if color is 0 (Red), bit is 1. If color is 1 (Blue), bit is 0.
        
        R = 0  # Bitset of Red triples
        B = 0  # Bitset of Blue triples
        
        if initial_colors is None:
            for i in range(num_triples):
                if random.random() < 0.5:
                    R |= (1 << i)
                else:
                    B |= (1 << i)
        else:
            for i, c in enumerate(initial_colors):
                if c == 0: # Red
                    R |= (1 << i)
                else: # Blue
                    B |= (1 << i)

        # Precompute violation status
        # red_viol[i] is True if triple i is part of a Red K4
        # blue_viol[i] is True if triple i is part of a Blue K5
        red_viol = [False] * num_triples
        blue_viol = [False] * num_triples
        
        for i in range(num_triples):
            if R & (1 << i):
                if (R & red_k4_masks[i]) == red_k4_masks[i]:
                    red_viol[i] = True
            else:
                if (B & blue_k5_masks[i]) == blue_k5_masks[i]:
                    blue_viol[i] = True
        
        current_cost = sum(red_viol) + sum(blue_viol)
        best_cost = current_cost
        best_R = R
        best_B = B
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        max_iters = 10000000
        
        for i in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            t_idx = random.randint(0, num_triples - 1)
            is_red = bool(R & (1 << t_idx))
            
            delta_cost = 0
            
            if is_red:
                # Flip Red -> Blue
                # 1. Update t_idx status
                if red_viol[t_idx]:
                    delta_cost -= 1
                    red_viol[t_idx] = False
                # Check if it becomes Blue violation
                if (B & blue_k5_masks[t_idx]) == blue_k5_masks[t_idx]:
                    if not blue_viol[t_idx]:
                        delta_cost += 1
                        blue_viol[t_idx] = True

                # 2. Update neighbors in red_k4_rev[t_idx]
                # These neighbors j are Red (since t_idx was Red, and j is in red_k4_rev, 
                # wait, red_k4_rev contains ALL j such that t is in mask[j]. 
                # If j is Blue, it doesn't care about t being Red/Blue for Red violations.
                # But if j is Red, its Red violation status depends on t.
                for j in red_k4_rev[t_idx]:
                    if R & (1 << j): # j is Red
                        # t was Red, now Blue.
                        # If j was a violation, it might cease to be one.
                        if red_viol[j]:
                            # Check if j is still a violation
                            # (R without t) & mask[j] == mask[j]?
                            # R has t set. We are flipping it.
                            # So we check (R & ~ (1<<t_idx) & red_k4_masks[j]) == red_k4_masks[j]
                            # Since mask[j] does not contain j itself, and contains t_idx,
                            # if t_idx was the "missing" red bit, it fails.
                            # Actually, simpler: 
                            # If red_viol[j] was True, it means all bits in mask[j] were set in R.
                            # One of those bits is t_idx.
                            # Now t_idx is not set in R (effectively).
                            # So red_viol[j] MUST become False.
                            red_viol[j] = False
                            delta_cost -= 1
                            # It cannot become True immediately because we removed a bit.
                            
                # 3. Update neighbors in blue_k5_rev[t_idx]
                # These neighbors j are Blue.
                # t was Red, now Blue.
                # If j was not a violation, it might become one.
                for j in blue_k5_rev[t_idx]:
                    if B & (1 << j): # j is Blue
                        # t was Red, now Blue.
                        # If j was a violation, it stays a violation (more blue bits).
                        # If j was NOT a violation, it might become one.
                        if not blue_viol[j]:
                            # Check if all bits in mask[j] are now Blue
                            # B has t set (after flip).
                            # So check (B & blue_k5_masks[j]) == blue_k5_masks[j]
                            # Note: B currently has t_idx set? 
                            # We haven't updated R and B yet!
                            # We are calculating delta.
                            # Let's assume the flip happens.
                            # New B has t_idx set.
                            # So we check if (B | (1<<t_idx)) & mask[j] == mask[j]
                            # Since t_idx is in mask[j], this is equivalent to 
                            # (B & (mask[j] ^ (1<<t_idx))) == (mask[j] ^ (1<<t_idx))
                            # i.e., all OTHER bits are Blue.
                            other_mask = blue_k5_masks[j] & ~(1 << t_idx)
                            if (B & other_mask) == other_mask:
                                blue_viol[j] = True
                                delta_cost += 1

                # Apply flip
                R &= ~(1 << t_idx)
                B |= (1 << t_idx)

            else:
                # Flip Blue -> Red
                # 1. Update t_idx status
                if blue_viol[t_idx]:
                    delta_cost -= 1
                    blue_viol[t_idx] = False
                if (R & red_k4_masks[t_idx]) == red_k4_masks[t_idx]:
                    if not red_viol[t_idx]:
                        delta_cost += 1
                        red_viol[t_idx] = True

                # 2. Update neighbors in red_k4_rev[t_idx]
                # t was Blue, now Red.
                # j is Red.
                for j in red_k4_rev[t_idx]:
                    if R & (1 << j): # j is Red
                        if not red_viol[j]:
                            other_mask = red_k4_masks[j] & ~(1 << t_idx)
                            if (R & other_mask) == other_mask:
                                red_viol[j] = True
                                delta_cost += 1

                # 3. Update neighbors in blue_k5_rev[t_idx]
                # t was Blue, now Red.
                # j is Blue.
                for j in blue_k5_rev[t_idx]:
                    if B & (1 << j): # j is Blue
                        if blue_viol[j]:
                            blue_viol[j] = False
                            delta_cost -= 1

                # Apply flip
                B &= ~(1 << t_idx)
                R |= (1 << t_idx)

            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_R = R
                best_B = B
                if best_cost == 0:
                    # Convert to list
                    colors = [0 if (best_R & (1 << i)) else 1 for i in range(num_triples)]
                    return colors, 0
            
            T *= cooling
            if T < T_min:
                T = T_min
                if i % 10000 == 0:
                    T = 10.0

        colors = [0 if (best_R & (1 << i)) else 1 for i in range(num_triples)]
        return colors, best_cost

    # Strategy: Try increasing n
    targets = [34, 33, 32, 31, 30, 29, 28, 27]
    time_allocations = [10, 10, 8, 5, 5, 3, 3, 3]
    
    best_found = (9, mod_color(9, 3))
    
    for i, n in enumerate(targets):
        if time.time() > deadline - 10:
            break
            
        alloc = time_allocations[i] if i < len(time_allocations) else 3
        current_deadline = time.time() + alloc
        
        max_restarts = 10
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline:
                break
                
            if r == 0:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 1:
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 2:
                init = [0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r == 3:
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            else:
                triple_list_n = list(combinations(range(n), 3))
                init = [0 if random.random() < 0.4 else 1 for _ in triple_list_n]

            colors, cost = run_bitwise_search(n, current_deadline, init)
            
            if cost == 0:
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
