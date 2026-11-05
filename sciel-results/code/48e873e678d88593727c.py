import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    def is_valid_addition(cand, family):
        """Check if cand can be added to family without creating a sunflower."""
        groups = {}
        for existing in family:
            inter = cand & existing
            key = frozenset(inter)
            if key not in groups:
                groups[key] = []
            groups[key].append(existing)
        
        for S, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                mi = members[i]
                for j in range(i+1, len(members)):
                    if mi & members[j] == S:
                        return False
        return True
    
    best_family = []
    
    rng = random.Random(42)
    
    # Try multiple greedy orderings
    num_orders = 15
    for trial in range(num_orders):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_sets))
        else:
            order = list(range(n_sets))
            rng.shuffle(order)
        
        family = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            cand = all_sets[idx]
            if is_valid_addition(cand, family):
                family.append(cand)
        
        if len(family) > len(best_family):
            best_family = family
    
    # Local search: try to improve
    if time.time() < deadline - 10:
        family = list(best_family)
        family_set = set(frozenset(s) for s in family)
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Try to add a random set
            improved = False
            for _ in range(100):
                idx = rng.randint(0, n_sets - 1)
                cand = all_sets[idx]
                if frozenset(cand) not in family_set:
                    if is_valid_addition(cand, family):
                        family.append(cand)
                        family_set.add(frozenset(cand))
                        improved = True
                        break
            
            if not improved:
                break
    
    return [sorted(s) for s in family]
