import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n_max = 192
    max_size = 48
    
    best_set = []
    best_size = 0
    
    def count_violations(S):
        """
        Counts the number of pairs (a, {b,c,d}) such that a in S, {b,c,d} subset S\{a},
        distinct, and a divides b+c+d.
        """
        if len(S) < 4:
            return 0
            
        S_list = list(S)
        N = len(S_list)
        violations = 0
        
        # For each a in S, count triples in S\{a} that sum to multiple of a
        for i, a in enumerate(S_list):
            # Get others
            others = S_list[:i] + S_list[i+1:]
            if len(others) < 3:
                continue
            
            # Count triples (b,c,d) in others such that (b+c+d) % a == 0
            # Optimization: We can break early if we just need to know if it's 0 or not?
            # No, we need the count for SA.
            
            # To speed up, we can precompute sums?
            # For fixed a, we need count of (b,c,d) in others with b+c+d = k*a.
            
            # Simple loop
            # Others size M = N-1
            # Triplets: M choose 3
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    violations += 1
                    
        return violations

    def try_grow(S):
        """
        Greedily tries to add elements to S while maintaining 0 violations.
        Returns the new set.
        """
        S_list = list(S)
        current_violations = count_violations(S_list)
        if current_violations != 0:
            return S_list
            
        # Try to add elements
        # Sort candidates by some heuristic? e.g. larger numbers first?
        # Or random?
        # Let's try all candidates not in S
        candidates = [x for x in range(1, n_max + 1) if x not in S_list]
        
        # Shuffle to avoid bias
        random.shuffle(candidates)
        
        for x in candidates:
            new_S = S_list + [x]
            # Check if new_S has 0 violations
            # We only need to check violations involving x?
            # No, adding x might create new triples for existing a's?
            # Yes.
            # But if S was valid, only new violations involve x.
            # 1. x divides sum of 3 in S_list
            # 2. a in S_list divides sum of (2 from S_list + x)
            
            # Check 1: x | sum(3 from S_list)
            valid = True
            if len(S_list) >= 3:
                for t in combinations(S_list, 3):
                    if sum(t) % x == 0:
                        valid = False
                        break
            
            if valid:
                # Check 2: a | (s1 + s2 + x) for a in S_list, s1,s2 in S_list\{a}
                for a in S_list:
                    # s1, s2 from S_list \ {a}
                    # If len(S_list \ {a}) < 2, skip
                    if len(S_list) - 1 >= 2:
                        others = [s for s in S_list if s != a]
                        for s1, s2 in combinations(others, 2):
                            if (s1 + s2 + x) % a == 0:
                                valid = False
                                break
                    if not valid:
                        break
                        
            if valid:
                S_list.append(x)
                if len(S_list) >= max_size:
                    break
                    
        return S_list

    # Main SA Loop
    # We run multiple restarts
    for restart in range(50): # Max restarts, break on time
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random set
        # Size between 40 and 48
        init_size = random.randint(40, 48)
        S = random.sample(range(1, n_max + 1), init_size)
        
        current_violations = count_violations(S)
        
        # If we start valid, try to grow
        if current_violations == 0:
            S_grown = try_grow(S)
            if len(S_grown) > best_size:
                best_set = S_grown
                best_size = len(S_grown)
                if best_size == max_size:
                    return best_set
            continue # Restart with new random set to explore more
        
        # SA Parameters
        T = 10.0
        min_T = 0.01
        cooling = 0.995
        max_steps = 1000 # Limit steps per restart to ensure we restart often
        
        S_set = set(S)
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return best_set if best_set else S # Return best found
            
            # Generate Move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                candidates = [x for x in range(1, n_max + 1) if x not in S_set]
                if not candidates: continue
                x = random.choice(candidates)
                new_S = S + [x]
                new_S_set = S_set | {x}
                
            elif move_type < 0.7:
                # Remove
                if not S: continue
                y = random.choice(S)
                new_S = [x for x in S if x != y]
                new_S_set = S_set - {y}
                
            else:
                # Swap
                if not S: continue
                candidates = [x for x in range(1, n_max + 1) if x not in S_set]
                if not candidates: continue
                y = random.choice(S)
                x = random.choice(candidates)
                new_S = [v if v != y else x for v in S]
                new_S_set = S_set - {y} | {x}
            
            # Calculate New Violations
            # Optimization: If the move is small, can we update violations incrementally?
            # Hard to implement correctly. Just recalculate.
            new_violations = count_violations(new_S)
            
            delta = new_violations - current_violations
            
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                S = new_S
                S_set = new_S_set
                current_violations = new_violations
                
                if current_violations == 0:
                    # Try to grow
                    S_grown = try_grow(S)
                    if len(S_grown) > best_size:
                        best_set = S_grown
                        best_size = len(S_grown)
                        if best_size == max_size:
                            return best_set
                    # Reset S to grown version to continue from there?
                    # Or restart SA?
                    # Let's update S to the grown one and continue SA (might shrink then grow)
                    S = S_grown
                    S_set = set(S)
                    current_violations = 0
                    
            T *= cooling
            if T < min_T:
                break
        
    return best_set
