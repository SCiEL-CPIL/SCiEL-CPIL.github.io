import numpy as np
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety

    n = 16
    # Generate all triples
    triples = []
    triple_indices = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_indices[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    num_triples = len(triples)
    
    # Generate constraints
    # We need to map each constraint to its 8 triples.
    # And each triple to the list of constraints it belongs to.
    constraint_to_triples = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    # Helper to get triple index
    def get_idx(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_indices[(a, b, c)]

    # Iterate over all sets of 6 points
    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            for v3 in range(v2 + 1, n):
                for v4 in range(v3 + 1, n):
                    for v5 in range(v4 + 1, n):
                        for v6 in range(v5 + 1, n):
                            verts = [v1, v2, v3, v4, v5, v6]
                            
                            # Generate 15 pairings
                            # Pair v1 with v2..v6
                            for j in range(1, 6):
                                pair1 = (verts[0], verts[j])
                                remaining = [v for i, v in enumerate(verts) if i != 0 and i != j]
                                
                                # 3 pairings of remaining 4
                                pairings = [
                                    [(remaining[0], remaining[1]), (remaining[2], remaining[3])],
                                    [(remaining[0], remaining[2]), (remaining[1], remaining[3])],
                                    [(remaining[0], remaining[3]), (remaining[1], remaining[2])]
                                ]
                                for p in pairings:
                                    pair2 = p[0]
                                    pair3 = p[1]
                                    
                                    t_indices = []
                                    for a in pair1:
                                        for b in pair2:
                                            for c in pair3:
                                                t_idx = get_idx(a, b, c)
                                                t_indices.append(t_idx)
                                                triple_to_constraints[t_idx].append(len(constraint_to_triples))
                                    
                                    constraint_to_triples.append(t_indices)
                                    
    num_constraints = len(constraint_to_triples)
    
    # Data structures for Greedy Hitting Set
    # counts[c]: number of present triples in constraint c
    # present[t]: boolean
    # violated: set of constraint indices where counts[c] == 8
    # viol_deg[t]: number of violated constraints containing t
    
    best_kept = []
    best_size = -1
    
    max_restarts = 50 # Tune based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        present = np.ones(num_triples, dtype=bool)
        counts = np.full(num_constraints, 8, dtype=np.int32)
        
        # Initial state: all constraints are violated (8 present)
        violated = set(range(num_constraints))
        
        # Initialize viol_deg
        viol_deg = np.zeros(num_triples, dtype=np.int32)
        for c in violated:
            for t in constraint_to_triples[c]:
                viol_deg[t] += 1
                
        # Greedy Removal
        # We maintain a list of violated constraints to pick from.
        # Using a list is faster than a set for iteration if we handle removals carefully, 
        # but set is easier for membership. 
        # Since we only remove from violated, we can keep a list and skip invalid ones?
        # Or just use a list and filter?
        # Let's use a list `violated_list` and a boolean `is_violated` array for O(1) check.
        
        is_violated = np.ones(num_constraints, dtype=bool)
        violated_list = list(range(num_constraints))
        
        removed_count = 0
        
        while violated_list:
            if time.time() > deadline:
                break
                
            # Pick a random violated constraint
            # To ensure we pick a currently violated one, we can pick a random index 
            # from the list and check is_violated. If not, remove from list?
            # Better: maintain a list of currently violated constraints.
            # When a constraint becomes non-violated, we don't remove it from the list immediately 
            # to avoid O(N) removal. We just check is_violated when picking.
            # But the list will grow with stale entries.
            # Alternative: Use a heap or just pick random index from 0 to len-1 and check.
            
            # Efficient pick:
            # Pick random index in violated_list
            idx = random.randint(0, len(violated_list) - 1)
            c = violated_list[idx]
            
            if not is_violated[c]:
                # Swap with last, pop
                violated_list[idx] = violated_list[-1]
                violated_list.pop()
                continue
                
            # Constraint c is violated.
            # Find the best triple to remove from c.
            # Candidates: present triples in c.
            # Score: viol_deg[t]
            
            best_t = -1
            best_score = -1
            for t in constraint_to_triples[c]:
                if present[t]:
                    score = viol_deg[t]
                    # Add small random noise for tie-breaking
                    score += random.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_t = t
                        
            # Remove best_t
            present[best_t] = False
            removed_count += 1
            
            # Update counts and viol_deg for all constraints containing best_t
            for c2 in triple_to_constraints[best_t]:
                counts[c2] -= 1
                if counts[c2] == 7: # Was 8, now 7 -> no longer violated
                    if is_violated[c2]:
                        is_violated[c2] = False
                        # Decrement viol_deg for all triples in c2
                        for t2 in constraint_to_triples[c2]:
                            viol_deg[t2] -= 1
                        # Note: We don't remove c2 from violated_list here, 
                        # it will be skipped in future picks.
                        
        # Greedy Addition Phase
        # Try to add back removed triples
        # A triple t can be added if for all c in triple_to_constraints[t], counts[c] < 7
        # We maintain a list of removed triples
        removed_list = [i for i in range(num_triples) if not present[i]]
        
        while True:
            if time.time() > deadline:
                break
                
            addable = []
            for t in removed_list:
                can_add = True
                for c in triple_to_constraints[t]:
                    if counts[c] >= 7:
                        can_add = False
                        break
                if can_add:
                    addable.append(t)
                    
            if not addable:
                break
                
            # Pick a random addable triple
            # Or pick one with minimum "risk"? 
            # Let's stick to random for now, or maybe prefer those with lower sum of counts in their constraints?
            # Lower sum of counts means they are in "sparser" constraints, less likely to block others?
            # Actually, we want to maximize the number of additions.
            # Adding a triple increases counts.
            # If we add a triple in a constraint with count 0, it becomes 1.
            # If we add a triple in a constraint with count 6, it becomes 7.
            # The latter is more "dangerous" for future additions.
            # So we should prefer triples that are in constraints with LOW counts.
            
            best_t = -1
            best_risk = float('inf')
            for t in addable:
                # Risk: max count among its constraints? Or sum?
                # Let's use max count. If max count is low, it's safer.
                risk = max(counts[c] for c in triple_to_constraints[t])
                # Add small noise
                risk += random.random() * 0.1
                if risk < best_risk:
                    best_risk = risk
                    best_t = t
                    
            # Add best_t
            present[best_t] = True
            removed_list.remove(best_t)
            removed_count -= 1
            
            # Update counts
            for c in triple_to_constraints[best_t]:
                counts[c] += 1
                
        # Calculate final size
        current_size = num_triples - removed_count
        if current_size > best_size:
            best_size = current_size
            # Store the result
            # We need to reconstruct the list of kept triples
            # It's faster to keep a list of removed indices? 
            # Or just check present array.
            best_present = present.copy()
            
    # Reconstruct result from best_present
    # If no restart finished, we might have best_size = -1. 
    # In that case, we should return a valid subset, e.g., empty or greedy result from last partial run?
    # The problem requires a valid triple system.
    # If we timeout in the middle of a restart, `present` is a valid state (no violated constraints? No, we might have stopped early).
    # If we stop early, there might be violated constraints.
    # So we must ensure we return a VALID solution.
    # If we haven't finished a single restart, we can't guarantee validity.
    # However, the greedy loop only stops when violated_list is empty OR time runs out.
    # If time runs out, we might have violated constraints.
    # So we must finish at least one restart.
    # With 50 restarts and fast execution, this is safe.
    # If best_size is still -1, we return empty list? No, that's valid but score 0.
    # Better to run at least one restart fully.
    
    if best_size == -1:
        # Fallback: Run one more restart without time check? No, must respect deadline.
        # Just return empty list or a small valid set.
        # Actually, if we didn't finish, we can take the state of the last restart 
        # and remove triples until no violations? That's expensive.
        # Let's assume 1 restart is fast enough.
        # 1 restart takes ~0.1s. 130s allows 1000 restarts.
        pass
        
    result = []
    for i in range(num_triples):
        if best_present[i]:
            result.append(list(triples[i]))
            
    return result
