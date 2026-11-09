"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    N = 4096
    MAX_SIZE = 80

    def build_greedy(start_from=1, max_size=MAX_SIZE, step=1):
        """Greedy: add numbers in order, maintaining set of triple sums."""
        selected = []
        triple_sums = set()
        
        for x in range(start_from, N + 1, step):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 10:
                break
            
            # Check if x can be added: all new triple sums involving x must be unique
            # New triples: (x,x,x), (x,x,y), (x,y,y), (x,y,z) for y,z in selected with y<=z
            # Actually need to check all triples from combinations_with_replacement(selected+[x], 3)
            # that include x at least once, and verify they don't conflict with existing sums
            # and are unique among themselves.
            
            new_sums = set()
            conflict = False
            
            # (x, x, x)
            s = 3 * x
            if s in triple_sums:
                conflict = True
            else:
                new_sums.add(s)
            
            if not conflict:
                # (x, x, y) for y in selected
                for y in selected:
                    s = 2 * x + y
                    if s in triple_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.add(s)
            
            if not conflict:
                # (x, y, y) for y in selected
                for y in selected:
                    s = x + 2 * y
                    if s in triple_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.add(s)
            
            if not conflict:
                # (x, y, z) for y, z in selected with y <= z
                for i in range(len(selected)):
                    for j in range(i, len(selected)):
                        y = selected[i]
                        z = selected[j]
                        s = x + y + z
                        if s in triple_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.add(s)
                    if conflict:
                        break
            
            if not conflict:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected

    def local_swap_improve(selected, triple_sums, max_iter=500):
        """Try to improve by swapping elements or adding new ones via local search."""
        selected = list(selected)
        triple_sums = set(triple_sums)
        improved = True
        iters = 0
        
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() > DEADLINE - 5:
                break
            
            # Try to add a new element
            for x in range(1, N + 1):
                if x in selected:
                    continue
                if time.time() > DEADLINE - 5:
                    break
                
                # Check if x can be added
                new_sums = set()
                conflict = False
                
                s = 3 * x
                if s in triple_sums:
                    conflict = True
                else:
                    new_sums.add(s)
                
                if not conflict:
                    for y in selected:
                        s = 2 * x + y
                        if s in triple_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.add(s)
                
                if not conflict:
                    for y in selected:
                        s = x + 2 * y
                        if s in triple_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.add(s)
                
                if not conflict:
                    for i in range(len(selected)):
                        for j in range(i, len(selected)):
                            y = selected[i]
                            z = selected[j]
                            s = x + y + z
                            if s in triple_sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.add(s)
                        if conflict:
                            break
                
                if not conflict:
                    selected.append(x)
                    triple_sums.update(new_sums)
                    improved = True
                    break  # Restart search for next addition
            
            # Try swaps: remove one element, add another
            if not improved and len(selected) > 0:
                for i in range(len(selected)):
                    if time.time() > DEADLINE - 5:
                        break
                    removed = selected[i]
                    # Remove triples involving removed
                    new_triple_sums = set()
                    for t in combinations_with_replacement(selected, 3):
                        if removed not in t:
                            new_triple_sums.add(sum(t))
                    
                    selected.remove(removed)
                    
                    for x in range(1, N + 1):
                        if x in selected:
                            continue
                        if time.time() > DEADLINE - 5:
                            break
                        
                        new_sums = set()
                        conflict = False
                        
                        s = 3 * x
                        if s in new_triple_sums:
                            conflict = True
                        else:
                            new_sums.add(s)
                        
                        if not conflict:
                            for y in selected:
                                s = 2 * x + y
                                if s in new_triple_sums or s in new_sums:
                                    conflict = True
                                    break
                                new_sums.add(s)
                        
                        if not conflict:
                            for y in selected:
                                s = x + 2 * y
                                if s in new_triple_sums or s in new_sums:
                                    conflict = True
                                    break
                                new_sums.add(s)
                        
                        if not conflict:
                            for ii in range(len(selected)):
                                for jj in range(ii, len(selected)):
                                    y = selected[ii]
                                    z = selected[jj]
                                    s = x + y + z
                                    if s in new_triple_sums or s in new_sums:
                                        conflict = True
                                        break
                                    new_sums.add(s)
                                if conflict:
                                    break
                        
                        if not conflict:
                            selected.append(x)
                            new_triple_sums.update(new_sums)
                            triple_sums = new_triple_sums
                            improved = True
                            break
                    
                    if not improved:
                        selected.append(removed)
                        triple_sums = set()
                        for t in combinations_with_replacement(selected, 3):
                            triple_sums.add(sum(t))
        
        return selected, triple_sums

    # Start with greedy construction
    best_selected = build_greedy(start_from=1, max_size=MAX_SIZE, step=1)
    best_sums = set(sum(t) for t in combinations_with_replacement(best_selected, 3))
    
    # Try to improve with local swap search
    if time.time() < DEADLINE - 10:
        improved_selected, improved_sums = local_swap_improve(best_selected, best_sums, max_iter=200)
        if len(improved_selected) > len(best_selected):
            best_selected = improved_selected
            best_sums = improved_sums
    
    # Try greedy with different starting points if time allows
    for start in [2, 3, 5]:
        if time.time() > DEADLINE - 15:
            break
        candidate = build_greedy(start_from=start, max_size=MAX_SIZE, step=1)
        if len(candidate) > len(best_selected):
            best_selected = candidate
            best_sums = set(sum(t) for t in combinations_with_replacement(best_selected, 3))
            if time.time() < DEADLINE - 10:
                improved_selected, improved_sums = local_swap_improve(best_selected, best_sums, max_iter=100)
                if len(improved_selected) > len(best_selected):
                    best_selected = improved_selected
                    best_sums = improved_sums
    
    return best_selected
# EVOLVE-BLOCK-END
