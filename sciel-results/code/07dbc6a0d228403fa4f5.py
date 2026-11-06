import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin (90s limit)
    
    n = 192
    max_size = 48
    
    # Precompute: for each x in 1..n, the set of triple sums (a+b+c) where a,b,c are distinct
    # elements from 1..n, a<b<c, a,b,c != x, and x divides (a+b+c).
    # This is too large. Instead, precompute for each x and each possible sum s (3..3*192=576),
    # whether x | s.
    
    # For check 1 (candidate x divides sum of 3 from S): 
    #   Precompute triple_sums[S] as a set of all triple sums from S.
    #   Then check if any triple sum is divisible by x.
    #   But maintaining this dynamically is expensive.
    
    # For check 2 (for each a in S, a divides b+c+x for b,c in S\{a}):
    #   For each a, precompute the set of residues { (b+c) mod a : b,c in S\{a}, b!=c }
    #   Then x is bad if x mod a is in { (-s) mod a : s in that set }
    
    # Let's use a hybrid: precompute divisibility tables
    # div_table[x][s] = True if x | s, for x in 1..n, s in 0..3*n
    max_sum = 3 * n  # max possible sum of 3 elements
    div_table = [[False] * (max_sum + 1) for _ in range(n + 1)]
    for x in range(1, n + 1):
        for s in range(x, max_sum + 1, x):
            div_table[x][s] = True
    
    def can_add(selected, candidate):
        """Check if candidate can be added to selected without violating constraints."""
        sel = selected
        m = len(sel)
        
        # Check 1: candidate divides sum of any 3 distinct elements in selected
        if m >= 3:
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = sel[i] + sel[j]
                    for k in range(j + 1, m):
                        if div_table[candidate][s2 + sel[k]]:
                            return False
        
        # Check 2: for each a in selected, a divides (b+c+candidate) for distinct b,c in selected, b,c != a
        for idx_a, a in enumerate(sel):
            # For pairs (b,c) in sel where b != a, c != a, b != c
            others = [sel[i] for i in range(m) if i != idx_a]
            mo = len(others)
            if mo >= 2:
                for i in range(mo):
                    for j in range(i + 1, mo):
                        if div_table[a][others[i] + others[j] + candidate]:
                            return False
        
        return True
    
    def is_valid_set(s):
        """Verify that set s satisfies all constraints."""
        m = len(s)
        for idx_x, x in enumerate(s):
            # Check: x does not divide sum of any 3 distinct other elements
            others = [s[i] for i in range(m) if i != idx_x]
            mo = len(others)
            if mo >= 3:
                for i in range(mo):
                    for j in range(i + 1, mo):
                        s2 = others[i] + others[j]
                        for k in range(j + 1, mo):
                            if div_table[x][s2 + others[k]]:
                                return False
        return True
    
    # Greedy: try to build a large set
    best = []
    
    # Try descending order first (largest numbers first)
    for x in range(n, 0, -1):
        if time.time() > deadline - 3.0:
            break
        if can_add(best, x):
            best.append(x)
        if len(best) >= max_size:
            break
    
    current_best = len(best)
    best_set = list(best)
    
    # Try multiple random orderings with different biases
    if time.time() < deadline - 5.0:
        for restart in range(200):
            if time.time() > deadline - 3.0:
                break
            
            # Different ordering strategies
            if restart % 5 == 0:
                candidates = list(range(1, n + 1))
                random.shuffle(candidates)
            elif restart % 5 == 1:
                # Descending with random perturbation
                candidates = list(range(n, 0, -1))
                random.shuffle(candidates)
            elif restart % 5 == 2:
                # Prioritize larger numbers with bias
                candidates = sorted(range(1, n + 1), key=lambda x: random.random() * 0.3 + x / n * 0.7, reverse=True)
            elif restart % 5 == 3:
                # Random but exclude small numbers first
                candidates = list(range(n, 0, -1))
                # Add some small numbers randomly
                small = list(range(1, n // 2 + 1))
                random.shuffle(small)
                candidates.extend(small[:n // 4])
                random.shuffle(candidates)
            else:
                # Weighted random: larger numbers more likely
                weights = [i ** 2 for i in range(1, n + 1)]
                candidates = random.choices(range(1, n + 1), weights=weights, k=n)
                # Remove duplicates while preserving order
                seen = set()
                unique_candidates = []
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        unique_candidates.append(c)
                candidates = unique_candidates
            
            curr = []
            for x in candidates:
                if len(curr) >= max_size:
                    break
                if can_add(curr, x):
                    curr.append(x)
            
            if len(curr) > current_best:
                # Validate before accepting
                if is_valid_set(curr):
                    current_best = len(curr)
                    best_set = list(curr)
    
    # Local search: try to swap elements to improve
    if time.time() < deadline - 5.0 and current_best < max_size:
        best_set = list(best_set)
        
        for iteration in range(1000):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            # Try removing one element and adding another
            for i in range(len(best_set)):
                if time.time() > deadline - 3.0:
                    break
                temp_set = best_set[:i] + best_set[i + 1:]
                temp_set_set = set(temp_set)
                
                # Try adding a candidate not in the set, prioritizing larger numbers
                for x in range(n, 0, -1):
                    if x in temp_set_set:
                        continue
                    if can_add(temp_set, x):
                        new_set = temp_set + [x]
                        if len(new_set) > len(best_set):
                            if is_valid_set(new_set):
                                best_set = new_set
                                improved = True
                            break
                        elif len(new_set) == len(best_set):
                            # Same size, but maybe better for future additions
                            if min(new_set) > min(best_set):
                                if is_valid_set(new_set):
                                    best_set = new_set
                                    improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
    
    # Final validation
    if not is_valid_set(best_set):
        # Fallback: return empty or a known valid set
        return []
    
    return best_set[:max_size]
