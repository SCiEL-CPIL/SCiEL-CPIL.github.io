"""Improved program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def check_sidon(s):
        """Check if s is a Sidon set (all pair sums a+b with a<=b are unique)."""
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        return len(sums) == len(set(sums))
    
    def get_diffs(s):
        """Get nonzero differences of set s."""
        return set(b - a for a, b in combinations(s, 2) if b != a)
    
    def greedy_build(order, deadline):
        """Build two sets greedily following the given order."""
        A = []
        B = []
        A_sums = set()
        B_sums = set()
        A_diffs = set()
        B_diffs = set()
        
        for x in order:
            if time.time() > deadline:
                break
            # Try adding to A
            can_add_A = True
            # Check Sidon for A
            for a in A:
                if (a + x) in A_sums:
                    can_add_A = False
                    break
            if can_add_A:
                # Check difference disjointness
                for a in A:
                    d = abs(x - a)
                    if d in B_diffs:
                        can_add_A = False
                        break
            if can_add_A:
                # Update A
                new_A = A + [x]
                if check_sidon(new_A):
                    for a in A:
                        A_sums.add(a + x)
                    A_sums.add(2 * x)
                    for a in A:
                        A_diffs.add(abs(x - a))
                    A = new_A
                else:
                    can_add_A = False
            
            # Try adding to B
            if not can_add_A:
                can_add_B = True
                for b in B:
                    if (b + x) in B_sums:
                        can_add_B = False
                        break
                if can_add_B:
                    for b in B:
                        d = abs(x - b)
                        if d in A_diffs:
                            can_add_B = False
                            break
                if can_add_B:
                    new_B = B + [x]
                    if check_sidon(new_B):
                        for b in B:
                            B_sums.add(b + x)
                        B_sums.add(2 * x)
                        for b in B:
                            B_diffs.add(abs(x - b))
                        B = new_B
        
        return A, B
    
    def local_search(A, B, deadline):
        """Try to improve by swapping or adding elements."""
        A = list(A)
        B = list(B)
        A_sums = set()
        B_sums = set()
        A_diffs = set()
        B_diffs = set()
        
        for a in A:
            for b in A:
                if a <= b:
                    A_sums.add(a + b)
        for b in B:
            for c in B:
                if b <= c:
                    B_sums.add(b + c)
        A_diffs = get_diffs(A)
        B_diffs = get_diffs(B)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding random unused elements
            used = set(A) | set(B)
            unused = [x for x in range(1, n + 1) if x not in used]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline:
                    break
                # Try adding to A
                can_add = True
                for a in A:
                    if (a + x) in A_sums:
                        can_add = False
                        break
                    if abs(x - a) in B_diffs:
                        can_add = False
                        break
                if can_add:
                    if check_sidon(A + [x]):
                        A.append(x)
                        for a in A[:-1]:
                            A_sums.add(a + x)
                        A_sums.add(2 * x)
                        for a in A[:-1]:
                            A_diffs.add(abs(x - a))
                        improved = True
                        break
                
                # Try adding to B
                can_add = True
                for b in B:
                    if (b + x) in B_sums:
                        can_add = False
                        break
                    if abs(x - b) in A_diffs:
                        can_add = False
                        break
                if can_add:
                    if check_sidon(B + [x]):
                        B.append(x)
                        for b in B[:-1]:
                            B_sums.add(b + x)
                        B_sums.add(2 * x)
                        for b in B[:-1]:
                            B_diffs.add(abs(x - b))
                        improved = True
                        break
        
        return A, B
    
    # Multiple random orderings
    best_A, best_B = [], []
    best_size = 0
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Create random ordering with bias
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B = greedy_build(order, deadline)
        current_size = len(A) + len(B)
        
        if current_size > best_size:
            best_A, best_B = A, B
            best_size = current_size
        
        # Local search on the best found so far
        if time.time() < deadline - 5:
            A2, B2 = local_search(A, B, deadline)
            size2 = len(A2) + len(B2)
            if size2 > best_size:
                best_A, best_B = A2, B2
                best_size = size2
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [sorted(best_A), sorted(best_B)]
# EVOLVE-BLOCK-END
