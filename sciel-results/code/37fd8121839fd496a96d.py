import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def is_valid_set(S):
        """Check if set S is valid: no element divides sum of any 3 distinct others."""
        S_list = list(S)
        n = len(S_list)
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(n) if j != i]
            for triple in combinations(others, 3):
                s = sum(triple)
                if s % a == 0:
                    return False
        return True
    
    def count_violations(S):
        """Count number of violations in set S."""
        S_list = list(S)
        n = len(S_list)
        violations = 0
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(n) if j != i]
            for triple in combinations(others, 3):
                s = sum(triple)
                if s % a == 0:
                    violations += 1
        return violations
    
    def greedy_build(order):
        """Build a valid set greedily following the given order."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            candidate = selected + [x]
            # Check if adding x creates any violation
            valid = True
            # Check x doesn't divide sum of any 3 others in selected
            if len(selected) >= 3:
                for triple in combinations(selected, 3):
                    if sum(triple) % x == 0:
                        valid = False
                        break
            # Check no existing element divides sum involving x
            if valid and len(selected) >= 2:
                for a in selected:
                    # a might divide (x + b + c) for b,c in selected, b!=c, b!=a, c!=a
                    others_for_a = [v for v in selected if v != a]
                    # We need to check triples from (selected + [x]) that include x and 2 from selected
                    # Actually, we need to check all triples from candidate that don't include a
                    # Simpler: check all triples from candidate excluding a
                    for triple in combinations(candidate, 3):
                        if a not in triple:
                            if sum(triple) % a == 0:
                                valid = False
                                break
                    if not valid:
                        break
            if valid:
                selected.append(x)
        return selected
    
    def local_search(S, time_limit=5.0):
        """Try to improve set S via local search (add, remove, swap)."""
        S = list(S)
        best = list(S)
        best_size = len(S)
        start = time.time()
        
        while time.time() - start < time_limit and time.time() < deadline:
            improved = False
            
            # Try adding elements
            for x in range(192, 0, -1):
                if x in S:
                    continue
                candidate = S + [x]
                if is_valid_set(candidate):
                    S = candidate
                    best = list(S)
                    best_size = len(S)
                    improved = True
                    break
            
            if not improved:
                # Try removing one and adding two
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    S_without = S[:i] + S[i+1:]
                    for x1 in range(192, 0, -1):
                        if x1 in S_without:
                            continue
                        for x2 in range(192, 0, -1):
                            if x2 in S_without or x2 == x1:
                                continue
                            candidate = S_without + [x1, x2]
                            if len(candidate) > best_size:
                                if is_valid_set(candidate):
                                    S = candidate
                                    best = list(S)
                                    best_size = len(S)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Strategy 1: Greedy from large to small
    best_set = greedy_build(range(192, 0, -1))
    best_size = len(best_set)
    
    if time.time() < deadline - 10:
        # Local search on the greedy result
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Strategy 2: Try different orderings
    if time.time() < deadline - 20:
        # Try building from a subset that avoids small numbers
        for start_val in [96, 80, 64, 100, 110]:
            if time.time() > deadline - 5:
                break
            order = list(range(192, start_val - 1, -1))
            result = greedy_build(order)
            if len(result) > best_size:
                best_set = result
                best_size = len(result)
                # Local search
                if time.time() < deadline - 10:
                    improved = local_search(best_set)
                    if len(improved) > best_size:
                        best_set = improved
                        best_size = len(best_set)
    
    # Strategy 3: Randomized restarts
    if time.time() < deadline - 15:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Random order with bias toward larger numbers
            order = list(range(192, 0, -1))
            random.shuffle(order)
            # Bias: sort by a score that favors larger numbers and numbers with fewer divisors
            def score(x):
                # Numbers with few divisors are "safer"
                divs = 0
                for d in range(2, int(x**0.5) + 1):
                    if x % d == 0:
                        divs += 2
                    if x // d != d and x % (x//d) == 0:
                        divs += 1
                return x * 10 - divs
            order.sort(key=score, reverse=True)
            result = greedy_build(order)
            if len(result) > best_size:
                best_set = result
                best_size = len(result)
                if time.time() < deadline - 10:
                    improved = local_search(best_set)
                    if len(improved) > best_size:
                        best_set = improved
                        best_size = len(best_set)
    
    # Final local search
    if time.time() < deadline - 3 and best_size < 48:
        improved = local_search(best_set, time_limit=min(10, deadline - time.time() - 2))
        if len(improved) > best_size:
            best_set = improved
    
    return best_set
