import time
import math
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    moduli = list(range(20, 201))
    
    def compat(m, r, placed):
        """Check if (m, r) is compatible with all placed moduli."""
        for m2, r2 in placed.items():
            if m == m2:
                continue
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def try_add(placed, m, deadline):
        """Try to add modulus m with the best residue. Returns (success, new_placed)."""
        if m in placed:
            return False, placed
        valid = []
        for r in range(m):
            if compat(m, r, placed):
                valid.append(r)
        if not valid:
            return False, placed
        # Pick residue that maximizes flexibility for remaining moduli
        best_r = valid[0]
        best_flex = -1
        for r in valid:
            flex = 0
            for m2 in moduli:
                if m2 in placed or m2 == m:
                    continue
                g = math.gcd(m, m2)
                rem = r % g
                if rem < m2:
                    count_bad = (m2 - 1 - rem) // g + 1
                else:
                    count_bad = 0
                flex += (m2 - count_bad)
            if flex > best_flex:
                best_flex = flex
                best_r = r
        new_placed = dict(placed)
        new_placed[m] = best_r
        return True, new_placed
    
    def score(placed):
        """Compute the reciprocal sum."""
        return sum(1.0 / m for m in placed.keys())
    
    def greedy_build(order, deadline):
        """Greedy construction following a given order of moduli."""
        placed = {}
        for m in order:
            if time.time() > deadline:
                break
            success, new_placed = try_add(placed, m, deadline)
            if success:
                placed = new_placed
        return placed
    
    def local_search(placed, deadline):
        """Perform local search with swap operations."""
        best = dict(placed)
        best_score = score(best)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            no_improve_count = 0
            keys = list(best.keys())
            
            # Try adding any missing modulus
            for m in moduli:
                if time.time() > deadline:
                    return best
                if m in best:
                    continue
                success, new_placed = try_add(best, m, deadline)
                if success:
                    new_score = score(new_placed)
                    if new_score > best_score + 1e-15:
                        best = new_placed
                        best_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
            
            if improved:
                continue
            
            # Try 1-for-1 swaps: remove one, add another
            for m_remove in keys:
                if time.time() > deadline:
                    return best
                temp = dict(best)
                del temp[m_remove]
                for m_add in moduli:
                    if m_add in temp:
                        continue
                    success, new_placed = try_add(temp, m_add, deadline)
                    if success:
                        new_score = score(new_placed)
                        if new_score > best_score + 1e-15:
                            best = new_placed
                            best_score = new_score
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swaps: remove two, add one
            for i in range(len(keys)):
                if time.time() > deadline:
                    return best
                m1 = keys[i]
                for j in range(i + 1, len(keys)):
                    m2 = keys[j]
                    temp = dict(best)
                    del temp[m1]
                    del temp[m2]
                    for m_add in moduli:
                        if m_add in temp:
                            continue
                        success, new_placed = try_add(temp, m_add, deadline)
                        if success:
                            new_score = score(new_placed)
                            if new_score > best_score + 1e-15:
                                best = new_placed
                                best_score = new_score
                                improved = True
                                no_improve_count = 0
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try 2-for-1 swaps: remove one, add two
            for m_remove in keys:
                if time.time() > deadline:
                    return best
                temp = dict(best)
                del temp[m_remove]
                added = 0
                for m_add in moduli:
                    if m_add in temp:
                        continue
                    success, new_temp = try_add(temp, m_add, deadline)
                    if success:
                        temp = new_temp
                        added += 1
                        if added == 2:
                            break
                if added == 2:
                    new_score = score(temp)
                    if new_score > best_score + 1e-15:
                        best = temp
                        best_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break
        
        return best
    
    def residue_polish(placed, deadline):
        """Try to improve residues for better flexibility."""
        best = dict(placed)
        best_score = score(best)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for m in list(best.keys()):
                if time.time() > deadline:
                    break
                old_r = best[m]
                # Try all valid residues
                best_flex = -1
                best_r = old_r
                for r in range(m):
                    if r == old_r:
                        continue
                    if compat(m, r, best):
                        # Compute flexibility
                        flex = 0
                        for m2 in moduli:
                            if m2 in best or m2 == m:
                                continue
                            g = math.gcd(m, m2)
                            rem = r % g
                            if rem < m2:
                                count_bad = (m2 - 1 - rem) // g + 1
                            else:
                                count_bad = 0
                            flex += (m2 - count_bad)
                        if flex > best_flex:
                            best_flex = flex
                            best_r = r
                if best_r != old_r:
                    # Check if this actually helps by trying to add more
                    temp = dict(best)
                    temp[m] = best_r
                    # Try to add missing moduli
                    added = 0
                    for m_add in moduli:
                        if m_add in temp:
                            continue
                        success, new_temp = try_add(temp, m_add, deadline)
                        if success:
                            temp = new_temp
                            added += 1
                    if added > 0:
                        new_score = score(temp)
                        if new_score > best_score + 1e-15:
                            best = temp
                            best_score = new_score
                            improved = True
        
        return best
    
    # Multi-start with different orderings
    best_result = {}
    best_final_score = 0.0
    
    # Start 1: ascending order
    placed1 = greedy_build(moduli, deadline)
    placed1 = local_search(placed1, deadline)
    placed1 = residue_polish(placed1, deadline)
    s1 = score(placed1)
    if s1 > best_final_score:
        best_final_score = s1
        best_result = placed1
    
    # Start 2: descending order
    placed2 = greedy_build(sorted(moduli, reverse=True), deadline)
    placed2 = local_search(placed2, deadline)
    placed2 = residue_polish(placed2, deadline)
    s2 = score(placed2)
    if s2 > best_final_score:
        best_final_score = s2
        best_result = placed2
    
    # Start 3: random order (several tries)
    random.seed(42)
    for trial in range(5):
        if time.time() > deadline:
            break
        order = moduli[:]
        random.shuffle(order)
        placed = greedy_build(order, deadline)
        placed = local_search(placed, deadline)
        placed = residue_polish(placed, deadline)
        s = score(placed)
        if s > best_final_score:
            best_final_score = s
            best_result = placed
    
    # Final polish with remaining time
    if time.time() < deadline:
        best_result = local_search(best_result, deadline)
        best_result = residue_polish(best_result, deadline)
    
    # Build result
    classes = [[m, best_result[m]] for m in sorted(best_result.keys())]
    
    # Compute score as float
    score_val = sum(Fraction(1, m) for m, r in classes)
    return classes

