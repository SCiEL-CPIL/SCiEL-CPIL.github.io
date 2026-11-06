"""Rewritten program for large four-uniform families without three sunflowers."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Generate all 4-sets from [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_sets = len(all_sets)  # 1820
    
    # Precompute pairwise intersection sizes as a 2D list
    inter_size = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        si = set(all_sets[i])
        for j in range(i+1, n_sets):
            sj = set(all_sets[j])
            isz = len(si & sj)
            inter_size[i][j] = isz
            inter_size[j][i] = isz
    
    # For a set c and a pair (x,y) in family, c creates a sunflower iff
    # inter_size[x][c] == inter_size[y][c] == inter_size[x][y]
    
    # Greedy with local search
    random.seed(123)
    best_family = []
    best_size = 0
    
    def try_greedy(order, max_iter=None):
        """Greedy: add sets in given order if valid"""
        family = []
        for idx in order:
            if max_iter and len(family) >= max_iter:
                break
            # Check if idx can be added
            valid = True
            for i in range(len(family)):
                x = family[i]
                for j in range(i+1, len(family)):
                    y = family[j]
                    if inter_size[x][idx] == inter_size[y][idx] == inter_size[x][y]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                family.append(idx)
        return family
    
    def local_search(family, max_iters=100):
        """Try to improve family by swapping or adding"""
        family = list(family)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding each non-family set
            family_set = set(family)
            for idx in range(n_sets):
                if idx in family_set:
                    continue
                # Check if idx can be added
                valid = True
                for i in range(len(family)):
                    x = family[i]
                    for j in range(i+1, len(family)):
                        y = family[j]
                        if inter_size[x][idx] == inter_size[y][idx] == inter_size[x][y]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
            # Try removing a set and adding another
            if not improved:
                for i in range(len(family)):
                    for idx in range(n_sets):
                        if idx in family_set:
                            continue
                        # Remove family[i], try adding idx
                        new_family = family[:i] + family[i+1:]
                        new_family_set = set(new_family)
                        valid = True
                        for a in range(len(new_family)):
                            x = new_family[a]
                            for b in range(a+1, len(new_family)):
                                y = new_family[b]
                                if inter_size[x][idx] == inter_size[y][idx] == inter_size[x][y]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            family = new_family + [idx]
                            family_set = set(family)
                            improved = True
                            break
                    if improved:
                        break
        return family
    
    # Try multiple greedy orderings
    orders = []
    orders.append(list(range(n_sets)))  # natural
    orders.append(list(range(n_sets-1, -1, -1)))  # reverse
    
    for _ in range(20):
        if time.time() > deadline:
            break
        o = list(range(n_sets))
        random.shuffle(o)
        orders.append(o)
    
    for order in orders:
        if time.time() > deadline:
            break
        family = try_greedy(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
    
    # Local search on best family
    if time.time() < deadline:
        improved_family = local_search(best_family, max_iters=50)
        if len(improved_family) > best_size:
            best_family = improved_family
            best_size = len(best_family)
    
    result = [sorted(all_sets[i]) for i in best_family]
    if not result:
        result = [sorted(all_sets[0])]
    
    return result
