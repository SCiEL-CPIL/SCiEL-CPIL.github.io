import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # 15 ways to partition 6 points into 3 pairs
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
    
    def time_left():
        return deadline - time.time()

    def is_valid_add(t_idx, counts):
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                return False
        return True

    def greedy_init():
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            if time_left() < 5:
                break
            if is_valid_add(t_idx, counts):
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    best_size = 0
    best_active = None
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.99995
    
    # Run multiple restarts
    num_restarts = 20 # We will break early based on time
    
    for r in range(num_restarts):
        if time_left() < 10:
            break
            
        active, counts, current_size = greedy_init()
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            
        T = T_start
        last_improve_time = time.time()
        
        # Candidate lists for efficiency
        # We will maintain these dynamically or recompute if they get stale?
        # Recomputing available list is O(N * K) which is ~5000 * 15 = 75k ops.
        # Doing it every step is too slow.
        # Strategy: 
        # 1. Try random add from all triples (low prob if dense)
        # 2. Better: Maintain a list of "available" triples.
        #    When we remove a triple, some blocked triples might become available.
        #    When we add a triple, some available triples might become blocked.
        
        # Let's stick to a simpler but effective heuristic:
        # 1. Randomly pick an active triple to remove.
        # 2. After removal, check if any *previously blocked* triple can now be added.
        #    To do this efficiently, we can just try a random number of inactive triples.
        #    Or, we can maintain a `blocked_by` count? No, counts are global.
        
        # Improved Move Strategy:
        # 1. With prob 0.5, try to ADD a random inactive triple.
        #    - If valid, accept (improvement).
        #    - If invalid, reject.
        # 2. With prob 0.5, try to REMOVE a random active triple.
        #    - Always accept removal? No, we want to maximize size.
        #    - We should only remove if we can replace it or if we are stuck.
        #    
        # Standard SA for MaxSAT/MaxCut style:
        # Move: Flip a bit.
        # Here: Add or Remove.
        # If Add is valid: Delta = +1. Always accept.
        # If Add is invalid: Delta = -inf (or large penalty). Reject.
        # If Remove: Delta = -1. Accept with prob exp(-1/T).
        
        # To escape local optima where no single add is valid, we need to remove one.
        # But removing one reduces size. We hope that removing one opens up multiple adds.
        
        while time_left() > 5:
            # 1. Try to add a random inactive triple
            if random.random() < 0.7: # Bias towards adding
                # Pick random inactive triple
                # Finding a random inactive triple quickly:
                # We can just pick random index and check if active. If active, skip or pick another?
                # Better: Maintain a list of inactive indices?
                # For simplicity, just pick random index.
                t_idx = random.randint(0, num_triples - 1)
                if not active[t_idx]:
                    if is_valid_add(t_idx, counts):
                        # Accept add
                        active[t_idx] = True
                        for k_idx in triple_to_k222s[t_idx]:
                            counts[k_idx] += 1
                        current_size += 1
                        last_improve_time = time.time()
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_active = active[:]
                    else:
                        # Invalid add, do nothing
                        pass
            else:
                # Try to remove a random active triple
                t_idx = random.randint(0, num_triples - 1)
                if active[t_idx]:
                    # Calculate acceptance probability for removal
                    # Delta = -1
                    # Prob = exp(-1/T)
                    if random.random() < (2.718281828 ** (-1.0 / T)):
                        # Accept removal
                        active[t_idx] = False
                        for k_idx in triple_to_k222s[t_idx]:
                            counts[k_idx] -= 1
                        current_size -= 1
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
                
            # If no improvement for a long time, restart?
            # Or just continue with low T (hill climbing).
            # Hill climbing with random restarts is often better.
            # Let's add a check: if no improvement in last 1 second, break inner loop and restart.
            if time.time() - last_improve_time > 1.0:
                break

    # Final Polish: Greedy Add
    if best_active:
        # Use the best solution found so far
        active = best_active[:]
        counts = [0] * num_k222s
        current_size = 0
        for i in range(num_triples):
            if active[i]:
                current_size += 1
                for k_idx in triple_to_k222s[i]:
                    counts[k_idx] += 1
        
        # Try to add any remaining valid triples
        changed = True
        while changed and time_left() > 2:
            changed = False
            for i in range(num_triples):
                if not active[i] and is_valid_add(i, counts):
                    active[i] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] += 1
                    changed = True
                    if current_size > best_size:
                        best_size = current_size
                        best_active = active[:]
        
        # Try to remove triples that are "expensive"? 
        # Actually, just returning the largest valid set is the goal.
        # The greedy add ensures we are maximal (no single add possible).
        
        result = []
        for i in range(num_triples):
            if best_active[i]:
                result.append(list(all_triples[i]))
        return result
    else:
        # Fallback
        active, counts, current_size = greedy_init()
        result = []
        for i in range(num_triples):
            if active[i]:
                result.append(list(all_triples[i]))
        return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
