import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256
    
    def get_diffs(s):
        """Get all nonzero differences for a set."""
        diffs = set()
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i+1, len(s_list)):
                d = s_list[j] - s_list[i]
                diffs.add(d)
                diffs.add(-d)
        return diffs
    
    def is_sidon(s):
        """Check if s is a Sidon set using differences."""
        s_list = list(s)
        diffs = set()
        for i in range(len(s_list)):
            for j in range(i+1, len(s_list)):
                d = s_list[j] - s_list[i]
                if d in diffs or -d in diffs:
                    return False
                diffs.add(d)
                diffs.add(-d)
        return True
    
    def can_add_A(x, A, diffs_A, diffs_B):
        """Check if x can be added to A."""
        A_list = list(A)
        new_diffs = set()
        for a in A_list:
            d = x - a
            if d in diffs_A or -d in diffs_A:
                return False
            if d in diffs_B or -d in diffs_B:
                return False
            new_diffs.add(d)
            new_diffs.add(-d)
        return True
    
    def can_add_B(x, B, diffs_A, diffs_B):
        """Check if x can be added to B."""
        B_list = list(B)
        new_diffs = set()
        for b in B_list:
            d = x - b
            if d in diffs_B or -d in diffs_B:
                return False
            if d in diffs_A or -d in diffs_A:
                return False
            new_diffs.add(d)
            new_diffs.add(-d)
        return True
    
    def greedy_one_ordering(order, deadline):
        """Single forward pass over the given ordering."""
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() > deadline:
                break
            # Try to add to A
            if len(A) < 64 and can_add_A(x, A, diffs_A, diffs_B):
                A.append(x)
                for a in A[:-1]:
                    d = x - a
                    diffs_A.add(d)
                    diffs_A.add(-d)
            # Try to add to B
            elif len(B) < 64 and can_add_B(x, B, diffs_A, diffs_B):
                B.append(x)
                for b in B[:-1]:
                    d = x - b
                    diffs_B.add(d)
                    diffs_B.add(-d)
        
        return A, B, diffs_A, diffs_B
    
    def local_search(A, B, diffs_A, diffs_B, deadline):
        """Improve via local search: try removing and re-adding, or swapping."""
        A = list(A)
        B = list(B)
        diffs_A = set(diffs_A)
        diffs_B = set(diffs_B)
        
        def rebuild_diffs_A():
            nonlocal diffs_A
            diffs_A = set()
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    d = A[j] - A[i]
                    diffs_A.add(d)
                    diffs_A.add(-d)
        
        def rebuild_diffs_B():
            nonlocal diffs_B
            diffs_B = set()
            for i in range(len(B)):
                for j in range(i+1, len(B)):
                    d = B[j] - B[i]
                    diffs_B.add(d)
                    diffs_B.add(-d)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add elements
            for x in range(1, n+1):
                if x in A or x in B:
                    continue
                if len(A) < 64 and can_add_A(x, A, diffs_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        d = x - a
                        diffs_A.add(d)
                        diffs_A.add(-d)
                    improved = True
                    break
                if len(B) < 64 and can_add_B(x, B, diffs_A, diffs_B):
                    B.append(x)
                    for b in B[:-1]:
                        d = x - b
                        diffs_B.add(d)
                        diffs_B.add(-d)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove an element from A, try to add to B or vice versa
            for i in range(len(A)):
                if time.time() > deadline:
                    return A, B
                x = A[i]
                A.remove(x)
                rebuild_diffs_A()
                # Try to add x to B
                if len(B) < 64 and can_add_B(x, B, diffs_A, diffs_B):
                    B.append(x)
                    for b in B[:-1]:
                        d = x - b
                        diffs_B.add(d)
                        diffs_B.add(-d)
                    improved = True
                    break
                # Try to add some other element to A
                found = False
                for y in range(1, n+1):
                    if y in A or y in B:
                        continue
                    if can_add_A(y, A, diffs_A, diffs_B):
                        A.append(y)
                        for a in A[:-1]:
                            d = y - a
                            diffs_A.add(d)
                            diffs_A.add(-d)
                        improved = True
                        found = True
                        break
                    if can_add_B(y, B, diffs_A, diffs_B) and len(B) < 64:
                        # This doesn't help A, but let's just try to add to A
                        pass
                if not found:
                    A.append(x)
                    for a in A[:-1]:
                        d = x - a
                        diffs_A.add(d)
                        diffs_A.add(-d)
                if improved:
                    break
            
            if improved:
                continue
            
            # Try removing from B and adding to A
            for i in range(len(B)):
                if time.time() > deadline:
                    return A, B
                x = B[i]
                B.remove(x)
                rebuild_diffs_B()
                if len(A) < 64 and can_add_A(x, A, diffs_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        d = x - a
                        diffs_A.add(d)
                        diffs_A.add(-d)
                    improved = True
                    break
                found = False
                for y in range(1, n+1):
                    if y in A or y in B:
                        continue
                    if can_add_B(y, B, diffs_A, diffs_B):
                        B.append(y)
                        for b in B[:-1]:
                            d = y - b
                            diffs_B.add(d)
                            diffs_B.add(-d)
                        improved = True
                        found = True
                        break
                if not found:
                    B.append(x)
                    for b in B[:-1]:
                        d = x - b
                        diffs_B.add(d)
                        diffs_B.add(-d)
                if improved:
                    break
        
        return A, B
    
    # Multi-start with different orderings
    best_A, best_B = [], []
    best_size = 0
    
    rng = random.Random(42)
    
    # Try several different orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Different strategies for ordering
        if trial == 0:
            order = list(range(1, n+1))
        elif trial == 1:
            order = list(range(n, 0, -1))
        elif trial == 2:
            order = list(range(1, n+1, 2)) + list(range(2, n+1, 2))
        elif trial == 3:
            order = list(range(2, n+1, 2)) + list(range(1, n+1, 2))
        elif trial == 4:
            order = list(range(1, n+1))
            rng.shuffle(order)
        else:
            order = list(range(1, n+1))
            rng.shuffle(order)
        
        A, B, diffs_A, diffs_B = greedy_one_ordering(order, deadline)
        
        # Local search
        A, B = local_search(A, B, diffs_A, diffs_B, deadline)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = list(A)
            best_B = list(B)
        
        if best_size >= 128:  # Max possible
            break
    
    return [best_A, best_B]
