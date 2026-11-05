import sys
import random

def construct():
    n = 256
    
    def is_valid_swap(S_set, S_list, new_elem, old_elem):
        """
        Check if replacing old_elem with new_elem in S results in a valid 3-AP free set.
        S_set: set of current elements
        S_list: list of current elements
        new_elem: element to add
        old_elem: element to remove
        """
        # Iterate over all a in S (excluding old_elem)
        for a in S_list:
            if a == old_elem:
                continue
            
            # a is in the new set
            
            # Case 1: new_elem is middle. a, new_elem, b => b = 2*new_elem - a
            b = 2 * new_elem - a
            # b must be in new set (S - {old_elem} + {new_elem})
            # b is in new set if b in S_set and b != old_elem
            # Also b must be distinct from a and new_elem?
            # AP terms must be distinct? "nonconstant" implies d != 0.
            # If d != 0, terms are distinct.
            # If b == a, then 2*new_elem = 2*a => new_elem = a.
            # But new_elem is not in S (old), and a is in S. So new_elem != a.
            # If b == new_elem, then 2*new_elem = a + new_elem => new_elem = a. Contradiction.
            # So distinctness is guaranteed if new_elem != a.
            
            if b in S_set and b != old_elem:
                return False

            # Case 2: new_elem is first. new_elem, a, b => b = 2*a - new_elem
            b = 2 * a - new_elem
            if b in S_set and b > a and b != old_elem:
                return False

            # Case 3: new_elem is last. a, b, new_elem => a = 2*b - new_elem => b = (a+new_elem)/2
            if (a + new_elem) % 2 == 0:
                b = (a + new_elem) // 2
                if b in S_set and b < a and b != old_elem:
                    return False
        
        return True

    def greedy_init():
        S = set()
        S_list = []
        for x in range(1, n + 1):
            valid = True
            # x is largest, so x is 3rd term: a, b, x => a = 2*b - x
            for b in S_list:
                a = 2 * b - x
                if a in S and a < b:
                    valid = False
                    break
            if valid:
                S.add(x)
                S_list.append(x)
        return S, S_list

    best_score = -1
    best_S = set()
    
    num_restarts = 15
    steps_per_restart = 30000
    
    for restart in range(num_restarts):
        random.seed(1000 + restart)
        
        S, S_list = greedy_init()
        rejected = [i for i in range(1, n + 1) if i not in S]
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        # Adjust steps to fit time
        # 15 restarts * 30000 steps = 450,000 steps.
        # Each step: O(N) check + O(N) update (remove from list).
        # 450,000 * 256 * 2 ~ 2.3 * 10^8 ops.
        # Might be too slow in Python (10-20 sec?).
        # Let's reduce steps or restarts.
        # 10 restarts, 20,000 steps.
        
        current_score = sum(1.0 / x for x in S)
        
        for step in range(steps_per_restart):
            # Pick random old_elem from S_list
            old_elem = random.choice(S_list)
            
            # Pick random new_elem from rejected
            if not rejected:
                break
            new_elem = random.choice(rejected)
            
            # Check validity
            if is_valid_swap(S, S_list, new_elem, old_elem):
                delta = (1.0 / new_elem) - (1.0 / old_elem)
                
                accept = False
                if delta > 0:
                    accept = True
                else:
                    if T > 0:
                        # Use math.exp for accuracy
                        prob = pow(2.718281828, delta / T)
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update structures
                    S.remove(old_elem)
                    S_list.remove(old_elem)
                    S.add(new_elem)
                    S_list.append(new_elem)
                    
                    rejected.remove(new_elem)
                    rejected.append(old_elem)
                    
                    current_score += delta
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if current_score > best_score:
            best_score = current_score
            best_S = S.copy()
    
    return sorted(best_S)
