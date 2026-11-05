import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    n = 256
    max_set_size = 64
    
    # Precompute conflict structures
    # For each value x in 1..n, and each other value y in 1..n,
    # we can precompute what constraints x+y and |x-y| would create
    
    # We'll use a different approach: model as assignment problem
    # Each element 1..256 can be: unassigned, in A, or in B
    # Constraints:
    #   - Within A: all pair sums (including doubles) unique
    #   - Within B: all pair sums (including doubles) unique
    #   - Nonzero differences of A and B must be disjoint
    
    # We'll use a greedy + local search approach with a different structure:
    # Instead of random walk, we'll do structured greedy with backtracking and 
    # then apply 2-opt style swaps
    
    # First, let's try multiple random seeds for greedy initialization
    best_size = 0
    best_A = []
    best_B = []
    
    def check_sidon(sums_set, diffs_set, A_list, B_list):
        """Verify the solution is valid"""
        # Check A is Sidon
        sums_A = set()
        diffs_A = set()
        for i, a in enumerate(A_list):
            if 2*a in sums_A:
                return False
            sums_A.add(2*a)
            for j in range(i):
                s = a + A_list[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
                d = abs(a - A_list[j])
                if d == 0:
                    continue
                if d in diffs_A:
                    return False
                diffs_A.add(d)
        
        sums_B = set()
        diffs_B = set()
        for i, b in enumerate(B_list):
            if 2*b in sums_B:
                return False
            sums_B.add(2*b)
            for j in range(i):
                s = b + B_list[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
                d = abs(b - B_list[j])
                if d == 0:
                    continue
                if d in diffs_B:
                    return False
                diffs_B.add(d)
        
        # Check disjoint differences
        if diffs_A & diffs_B:
            return False
        
        return True
    
    def greedy_build(order, bias_A=0.5):
        """Greedy build given an ordering of candidates"""
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        def can_add_A(x):
            if 2*x in sums_A:
                return False
            for a in A:
                if x + a in sums_A:
                    return False
                d = abs(x - a)
                if d in diffs_A or d in diffs_B:
                    return False
            return True
        
        def can_add_B(x):
            if 2*x in sums_B:
                return False
            for b in B:
                if x + b in sums_B:
                    return False
                d = abs(x - b)
                if d in diffs_B or d in diffs_A:
                    return False
            return True
        
        def do_add_A(x):
            A.append(x)
            sums_A.add(2*x)
            for a in A[:-1]:
                sums_A.add(x + a)
                d = abs(x - a)
                if d != 0:
                    diffs_A.add(d)
        
        def do_add_B(x):
            B.append(x)
            sums_B.add(2*x)
            for b in B[:-1]:
                sums_B.add(x + b)
                d = abs(x - b)
                if d != 0:
                    diffs_B.add(d)
        
        for x in order:
            if len(A) >= max_set_size and len(B) >= max_set_size:
                break
            # Decide where to try
            if len(A) < max_set_size and (len(B) >= max_set_size or random.random() < bias_A):
                if can_add_A(x):
                    do_add_A(x)
            elif len(B) < max_set_size:
                if can_add_B(x):
                    do_add_B(x)
            elif len(A) < max_set_size:
                if can_add_A(x):
                    do_add_A(x)
        
        return A, B
    
    # Try multiple greedy builds with different random orderings
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        bias = random.random() * 0.4 + 0.3  # 0.3 to 0.7
        A, B = greedy_build(order, bias)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = list(A)
            best_B = list(B)
    
    # Local search: try swaps to improve
    # Represent current state
    A = list(best_A)
    B = list(best_B)
    
    def rebuild_state(A, B):
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        for i, a in enumerate(A):
            sums_A.add(2*a)
            for j in range(i):
                sums_A.add(a + A[j])
                d = abs(a - A[j])
                if d != 0:
                    diffs_A.add(d)
        for i, b in enumerate(B):
            sums_B.add(2*b)
            for j in range(i):
                sums_B.add(b + B[j])
                d = abs(b - B[j])
                if d != 0:
                    diffs_B.add(d)
        return sums_A, sums_B, diffs_A, diffs_B
    
    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        if 2*x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            d = abs(x - a)
            if d in diffs_A or d in diffs_B:
                return False
        return True
    
    def can_add_to_B(x, B, sums_B, diffs_B, diffs_A):
        if 2*x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            d = abs(x - b)
            if d in diffs_B or d in diffs_A:
                return False
        return True
    
    # Local search with swap operations
    A_set = set(A)
    B_set = set(B)
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Try to add a random element
        progress = False
        x = random.randint(1, n)
        if x not in A_set and x not in B_set:
            sums_A, sums_B, diffs_A, diffs_B = rebuild_state(A, B)
            if len(A) < max_set_size and can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                A.append(x)
                A_set.add(x)
                progress = True
            elif len(B) < max_set_size and can_add_to_B(x, B, sums_B, diffs_B, diffs_A):
                B.append(x)
                B_set.add(x)
                progress = True
        
        # Try a swap: remove from A, add to B or vice versa
        if not progress and random.random() < 0.5:
            if A:
                i = random.randint(0, len(A)-1)
                x_remove = A[i]
                A.remove(x_remove)
                A_set.discard(x_remove)
                # Try to add a new element
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    y = random.randint(1, n)
                    if y in A_set or y in B_set:
                        continue
                    sums_A, sums_B, diffs_A, diffs_B = rebuild_state(A, B)
                    if len(A) < max_set_size and can_add_to_A(y, A, sums_A, diffs_A, diffs_B):
                        A.append(y)
                        A_set.add(y)
                        break
                    elif len(B) < max_set_size and can_add_to_B(y, B, sums_B, diffs_B, diffs_A):
                        B.append(y)
                        B_set.add(y)
                        break
        
        # Update best
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)
    
    # Final verification
    if not check_sidon(None, None, best_A, best_B):
        # Fallback: use a simple valid solution
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
