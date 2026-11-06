import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
        
    # 1. Initial Greedy Construction
    # Randomize order, add if valid
    def greedy_init():
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if time_left() < 5:
                break
            # Check validity
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
                    
        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # Run multiple SA restarts with swap-based moves
    # SA parameters
    T_start = 2.0
    alpha = 0.99995  # Slower cooling
    
    num_restarts = 15
    
    for r in range(num_restarts):
        if time_left() < 15:
            break
            
        active, counts, current_size = greedy_init()
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
            
        T = T_start
        no_improve = 0
        max_no_improve = 50000
        
        while time_left() > 5:
            # Choose move type: 70% swap, 30% add/remove
            move_type = random.random()
            
            if move_type < 0.7 and current_size > 0:
                # SWAP: remove one active triple, add one inactive triple
                # Pick random active triple to remove
                # Find an active triple efficiently
                # We'll try multiple times to find a valid swap
                
                # Pick a random active triple
                attempts = 0
                max_attempts = 20
                t_remove = -1
                t_add = -1
                
                while attempts < max_attempts and t_remove == -1:
                    t_remove = random.randint(0, num_triples - 1)
                    if not active[t_remove]:
                        t_remove = -1
                    attempts += 1
                
                if t_remove == -1:
                    continue
                    
                # Pick a random inactive triple to add
                attempts = 0
                while attempts < max_attempts and t_add == -1:
                    t_add = random.randint(0, num_triples - 1)
                    if active[t_add]:
                        t_add = -1
                    attempts += 1
                
                if t_add == -1:
                    continue
                    
                # Check if this swap is valid
                # After removing t_remove and adding t_add, no K222 should be complete
                # A K222 becomes complete if all 8 triples are present
                # We need to check K222s that contain both t_remove and t_add, 
                # or K222s that would become complete after the swap
                
                # Simpler: tentatively remove t_remove, then check if t_add can be added
                # Remove t_remove temporarily
                for k_idx in triple_to_k222s[t_remove]:
                    counts[k_idx] -= 1
                
                # Check if t_add is now valid
                valid_add = True
                for k_idx in triple_to_k222s[t_add]:
                    if counts[k_idx] == 7:
                        valid_add = False
                        break
                
                if valid_add:
                    # Accept the swap (size stays the same, but structure changes)
                    # This is always neutral in terms of size, so accept with probability
                    # based on whether it might lead to future improvements
                    # For now, always accept neutral swaps to explore the landscape
                    active[t_remove] = False
                    active[t_add] = True
                    for k_idx in triple_to_k222s[t_add]:
                        counts[k_idx] += 1
                    # Size unchanged
                else:
                    # Reject: restore counts
                    for k_idx in triple_to_k222s[t_remove]:
                        counts[k_idx] += 1
                
            else:
                # ADD or REMOVE move
                if random.random() < 0.5:
                    # Try to add
                    t_idx = random.randint(0, num_triples - 1)
                    if not active[t_idx]:
                        valid = True
                        for k_idx in triple_to_k222s[t_idx]:
                            if counts[k_idx] == 7:
                                valid = False
                                break
                        
                        if valid:
                            for k_idx in triple_to_k222s[t_idx]:
                                counts[k_idx] += 1
                            active[t_idx] = True
                            current_size += 1
                            no_improve = 0
                            if current_size > best_size:
                                best_size = current_size
                                best_active = active[:]
                                best_counts = counts[:]
                        else:
                            no_improve += 1
                else:
                    # Try to remove
                    t_idx = random.randint(0, num_triples - 1)
                    if active[t_idx]:
                        # Check if removal is beneficial (might allow future additions)
                        # For now, only remove if it leads to an immediate improvement possibility
                        # Or accept with some probability to escape local optima
                        remove_prob = min(0.3, 0.1 + 0.1 * (1.0 - current_size / 5000))
                        if random.random() < remove_prob:
                            for k_idx in triple_to_k222s[t_idx]:
                                counts[k_idx] -= 1
                            active[t_idx] = False
                            current_size -= 1
                            no_improve += 1
                        else:
                            no_improve += 1
            
            # Cool down
            T *= alpha
            
            # If stuck for too long, do a restart or perturbation
            if no_improve > max_no_improve:
                # Perturb: randomly remove some triples and re-add greedily
                if current_size > 10:
                    # Remove 5-10 random active triples
                    num_remove = random.randint(5, 10)
                    for _ in range(num_remove):
                        t_idx = random.randint(0, num_triples - 1)
                        if active[t_idx]:
                            for k_idx in triple_to_k222s[t_idx]:
                                counts[k_idx] -= 1
                            active[t_idx] = False
                            current_size -= 1
                    no_improve = 0
                    T = max(T, 0.5)  # Reheat
        
        # End of SA loop for this restart
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
            
    # Construct result from best_active
    if best_active is None:
        # Fallback
        active, counts, current_size = greedy_init()
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
