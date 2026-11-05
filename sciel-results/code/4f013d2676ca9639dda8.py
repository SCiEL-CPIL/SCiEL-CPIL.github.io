import time
import random
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (5s buffer)
    
    def get_triples(n):
        return list(combinations(range(n), 3))

    def precompute(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        # Map each triple to the 4-sets and 5-sets it belongs to
        # For efficiency, we store lists of indices
        triple_to_4sets = [[] for _ in range(len(triples))]
        triple_to_5sets = [[] for _ in range(len(triples))]
        
        for i, s4 in enumerate(four_sets):
            a, b, c, d = s4
            for t in combinations(s4, 3):
                triple_to_4sets[triple_index[t]].append(i)
                
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def initial_coloring(n, mode):
        """
        Generate structured initial colorings to avoid random traps.
        Mode 0: Random
        Mode 1: Modular (x+y+z mod 2)
        Mode 2: Finite field-like (x*y + y*z + z*x mod 2) 
        Mode 3: Threshold based on sum
        """
        colors = []
        for a, b, c in combinations(range(n), 3):
            if mode == 0:
                col = random.randint(0, 1)
            elif mode == 1:
                col = (a + b + c) % 2
            elif mode == 2:
                col = (a * b + b * c + c * a) % 2
            elif mode == 3:
                col = 0 if (a + b + c) < n else 1
            else:
                col = random.randint(0, 1)
            colors.append(col)
        return colors

    def solve_for_n(n, time_limit):
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        # Try multiple initializations
        best_result = None
        best_cost = float('inf')
        
        modes = [1, 2, 3, 0, 0, 0] # Prioritize structured
        random.shuffle(modes)
        
        for mode in modes:
            if time.time() > time_limit:
                break
                
            colors = initial_coloring(n, mode)
            
            # State: red_count_4, blue_count_5
            red_count_4 = [0] * num_4sets
            blue_count_5 = [0] * num_5sets
            
            # Initialize counts
            for t_idx in range(num_triples):
                is_red = (colors[t_idx] == 0)
                for s4_idx in triple_to_4sets[t_idx]:
                    if is_red:
                        red_count_4[s4_idx] += 1
                for s5_idx in triple_to_5sets[t_idx]:
                    if not is_red:
                        blue_count_5[s5_idx] += 1
            
            # Priority Queue of violations
            # (cost, type, index)
            # Type 0: Red K4 violation (red_count_4 == 4)
            # Type 1: Blue K5 violation (blue_count_5 == 10)
            pq = []
            
            for i in range(num_4sets):
                if red_count_4[i] == 4:
                    heapq.heappush(pq, (4, 0, i))
            
            for j in range(num_5sets):
                if blue_count_5[j] == 10:
                    heapq.heappush(pq, (10, 1, j))
            
            current_cost = len(pq)
            
            if current_cost == 0:
                return colors
            
            # Local Search Loop
            # We want to flip triples to reduce violations.
            # A flip changes counts of all 4-sets and 5-sets containing the triple.
            
            # To speed up, we can estimate the "potential" of a triple.
            # But let's stick to a greedy hill-climbing with random restarts within the time limit.
            
            steps = 0
            max_steps = 50000 # Hard cap to prevent infinite loops if stuck
            
            while time.time() < time_limit and steps < max_steps:
                if not pq:
                    break
                    
                # Get the most severe violation? Or just any?
                # Let's pick a random violation from the top 10 to diversify
                # Actually, just popping the smallest (which is 4 or 10) is fine, but let's randomize among violations.
                
                # Efficiently pick a random violation
                # Since pq might have stale entries, we need to be careful.
                # Simpler: maintain a list of current violations? 
                # For n=34, num_4sets ~ 46k, num_5sets ~ 237k. 
                # Re-scanning is O(N^5) which is too slow.
                
                # Let's use the PQ but handle staleness.
                # Entry: (cost, type, index)
                # We check if it's still a violation when we pop it.
                
                popped = None
                while pq:
                    cost, typ, idx = heapq.heappop(pq)
                    # Check if still valid
                    if typ == 0:
                        if red_count_4[idx] == 4:
                            popped = (typ, idx)
                            break
                    else:
                        if blue_count_5[idx] == 10:
                            popped = (typ, idx)
                            break
                
                if popped is None:
                    break # No more violations? Or all stale?
                    
                typ, idx = popped
                
                # Find a triple to flip within this set to reduce violation
                # If typ == 0 (Red K4): We need to flip one of the 4 red triples to blue.
                # If typ == 1 (Blue K5): We need to flip one of the 10 blue triples to red.
                
                if typ == 0:
                    # Get the 4 triples in this 4-set
                    s4 = four_sets[idx]
                    # The 4 triples are combinations of 3 from s4
                    candidates = []
                    for t in combinations(s4, 3):
                        t_idx = triple_index[t]
                        if colors[t_idx] == 0: # Is Red
                            candidates.append(t_idx)
                    
                    if not candidates:
                        continue # Should not happen if count was 4
                        
                    # Pick a candidate to flip
                    # Heuristic: Pick the one that causes least new damage?
                    # For now, random pick among candidates
                    t_idx = random.choice(candidates)
                    
                else:
                    s5 = five_sets[idx]
                    candidates = []
                    for t in combinations(s5, 3):
                        t_idx = triple_index[t]
                        if colors[t_idx] == 1: # Is Blue
                            candidates.append(t_idx)
                    
                    if not candidates:
                        continue
                        
                    t_idx = random.choice(candidates)
                
                # Perform Flip
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                # Update counts and PQ
                # We need to re-push affected sets if their status changes
                
                # Red count changes
                for s4_idx in triple_to_4sets[t_idx]:
                    old_c = red_count_4[s4_idx]
                    if old_color == 0: # R->B
                        red_count_4[s4_idx] = old_c - 1
                        # If it was 4, it's now 3 (fixed). If it was 3, now 2.
                        # If it was 4, we removed a violation.
                        # If it was 3 and becomes 4? No, R->B decreases red.
                        # So we only fix violations, don't create new red K4s.
                        # Wait, if old_c was 4, new is 3. Violation removed.
                        # If old_c was < 4, new is even lower. No new violation.
                    else: # B->R
                        red_count_4[s4_idx] = old_c + 1
                        # If old_c was 3, new is 4. New violation!
                        if old_c == 3:
                            heapq.heappush(pq, (4, 0, s4_idx))
                            current_cost += 1
                        # If old_c was 4, new is 5. Still a violation? 
                        # Definition: "No 4 vertices may have all four triples red".
                        # If count > 4, it's impossible for a 4-set (max 4 triples).
                        # So count can only be 0..4.
                        # So if old_c=4, it stays 4? No, max is 4.
                        # So B->R on a set with 3 reds makes it 4.
                        # B->R on a set with 4 reds? Impossible, all are red.
                        
                # Blue count changes
                for s5_idx in triple_to_5sets[t_idx]:
                    old_c = blue_count_5[s5_idx]
                    if old_color == 1: # B->R
                        blue_count_5[s5_idx] = old_c - 1
                        # If old_c was 10, new is 9. Fixed.
                    else: # R->B
                        blue_count_5[s5_idx] = old_c + 1
                        # If old_c was 9, new is 10. New violation!
                        if old_c == 9:
                            heapq.heappush(pq, (10, 1, s5_idx))
                            current_cost += 1
                        # Max blue is 10.
                        
                # Apply flip
                colors[t_idx] = new_color
                
                steps += 1
                
                # Periodically clean up PQ? 
                # The staleness check handles it, but PQ might grow large.
                # For n=34, this is manageable.
                
                if current_cost == 0:
                    return colors
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = colors.copy()
                
            if best_cost == 0:
                return best_result
                
        if best_result is not None:
            return best_result
        return None

    # Try n from 34 downwards
    # Allocate time based on difficulty? 
    # n=34 is the target. Give it the most time.
    # n=33, 32... give less.
    
    # Let's give 80s to n=34, 10s to 33, 5s to 32, etc.
    time_alloc = {
        34: 80.0,
        33: 10.0,
        32: 5.0,
        31: 2.0,
        30: 1.0
    }
    
    for n in range(34, 8, -1):
        if time.time() > deadline - 2.0:
            break
            
        t_limit = time_alloc.get(n, 1.0)
        t_start = time.time()
        
        # Cap t_limit by remaining time
        remaining = deadline - 2.0 - t_start
        if remaining <= 0:
            break
        t_limit = min(t_limit, remaining)
        
        res = solve_for_n(n, t_start + t_limit)
        if res is not None:
            # Verify validity before returning
            # (Optional but safe)
            if verify(n, res):
                return {'n': n, 'colors': res}
            else:
                # If invalid, continue to next n? Or try to fix?
                # The solver should return valid if cost 0.
                # If it returned non-zero cost, it's invalid.
                # My solver returns None if cost != 0 at the end of a mode, 
                # but keeps best_result. 
                # If best_cost > 0, it returns best_result which is invalid.
                # I need to ensure I only return valid ones.
                # Let's fix the return logic in solve_for_n.
                pass

    # Fallback
    n = 4
    colors = [0, 0, 0, 1]
    return {'n': n, 'colors': colors}

def verify(n, colors):
    """
    Verify if a coloring is valid.
    colors: list of 0/1 for all triples in lex order.
    """
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Check Red K4
    for s4 in combinations(range(n), 4):
        is_red = True
        for t in combinations(s4, 3):
            if colors[triple_index[t]] != 0:
                is_red = False
                break
        if is_red:
            return False
            
    # Check Blue K5
    for s5 in combinations(range(n), 5):
        is_blue = True
        for t in combinations(s5, 3):
            if colors[triple_index[t]] != 1:
                is_blue = False
                break
        if is_blue:
            return False
            
    return True
