import time
import random
from itertools import combinations_with_replacement, combinations

def _sidon_sums_ok(cand):
    """Check if all pair sums (with replacement) in cand are unique."""
    n = len(cand)
    if n < 2:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = cand[i] + cand[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diffs_of(cand):
    """Return set of nonzero differences b-a for a<b in cand."""
    n = len(cand)
    d = set()
    for i in range(n):
        for j in range(i + 1, n):
            d.add(cand[j] - cand[i])
    return d

def _try_construct(order, deadline):
    """Greedy construction given an element ordering. Returns (A, B) or (None, None)."""
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()
    
    for x in order:
        if time.time() > deadline:
            break
        # Try adding to A
        added = False
        # Check Sidon sums for A
        ok_sums = True
        for a in A:
            s = a + x
            if s in sumsA:
                ok_sums = False
                break
            s2 = x + x
            if s2 in sumsA and x != a:
                # Actually need to check all new sums
                pass
        # More careful check
        new_sums_A = set()
        for a in A:
            new_sums_A.add(a + x)
        new_sums_A.add(x + x)
        if new_sums_A & sumsA:
            ok_sums = False
        else:
            ok_sums = True
        
        if ok_sums:
            new_diffs_A = set()
            for a in A:
                new_diffs_A.add(x - a)
                new_diffs_A.add(a - x)
            # Only positive differences matter for the set, but let's use abs
            # Actually the problem says "nonzero difference sets" - let me use signed
            # Re-reading: "Their nonzero difference sets must be disjoint"
            # I'll use all nonzero differences (both signs)
            pos_diffs_A = set()
            for a in A:
                pos_diffs_A.add(x - a)
            neg_diffs_A = set()
            for a in A:
                neg_diffs_A.add(a - x)
            all_new_diffs_A = pos_diffs_A | neg_diffs_A
            
            if not (all_new_diffs_A & diffsB):
                # Can add to A
                A.append(x)
                sumsA |= new_sums_A
                diffsA |= all_new_diffs_A
                added = True
        
        # If not added to A, try B
        if not added:
            new_sums_B = set()
            for b in B:
                new_sums_B.add(b + x)
            new_sums_B.add(x + x)
            if new_sums_B & sumsB:
                ok_sums = False
            else:
                ok_sums = True
            
            if ok_sums:
                all_new_diffs_B = set()
                for b in B:
                    all_new_diffs_B.add(x - b)
                    all_new_diffs_B.add(b - x)
                
                if not (all_new_diffs_B & diffsA):
                    B.append(x)
                    sumsB |= new_sums_B
                    diffsB |= all_new_diffs_B
    
    return A, B

def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint difference sets."""
    if not A or not B:
        return False
    if len(A) > 64 or len(B) > 64:
        return False
    
    # Check Sidon property for A
    sumsA = set()
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
    
    # Check Sidon property for B
    sumsB = set()
    for i, b in enumerate(B):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
    
    # Check difference sets are disjoint
    diffsA = set()
    for i, a in enumerate(A):
        for j in range(i + 1, len(A)):
            diffsA.add(A[j] - A[i])
            diffsA.add(A[i] - A[j])
    
    diffsB = set()
    for i, b in enumerate(B):
        for j in range(i + 1, len(B)):
            diffsB.add(B[j] - B[i])
            diffsB.add(B[i] - B[j])
    
    if diffsA & diffsB:
        return False
    
    return True

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    deadline = start_time + time_limit
    
    n = 256
    best_A = [1]
    best_B = [2]
    best_size = 2
    
    # Initial simple construction
    A, B = _try_construct(list(range(1, n + 1)), deadline)
    if A and B and len(A) + len(B) > best_size:
        if _validate(A, B):
            best_A, best_B, best_size = A, B, len(A) + len(B)
    
    # Randomized restarts with different orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Different strategies for ordering
        strategy = random.randint(0, 4)
        
        if strategy == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 1:
            order = list(range(1, n + 1))
            random.shuffle(order)
            # Put smaller elements first
            order.sort(key=lambda x: random.random())
        elif strategy == 2:
            # Alternate small and large
            small = list(range(1, n // 2 + 1))
            large = list(range(n // 2 + 1, n + 1))
            random.shuffle(small)
            random.shuffle(large)
            order = []
            i = j = 0
            while i < len(small) or j < len(large):
                if i < len(small):
                    order.append(small[i])
                    i += 1
                if j < len(large):
                    order.append(large[j])
                    j += 1
        elif strategy == 3:
            # Random subset then fill in
            order = list(range(1, n + 1))
            random.shuffle(order)
        else:
            # Sorted with random perturbations
            order = list(range(1, n + 1))
            for i in range(1, len(order)):
                if random.random() < 0.3:
                    swap = random.randint(0, i)
                    order[i], order[swap] = order[swap], order[i]
        
        A, B = _try_construct(order, deadline)
        if A and B:
            size = len(A) + len(B)
            if size > best_size:
                if _validate(A, B):
                    best_A, best_B, best_size = A, B, size
    
    # Final polish: try to add more elements
    if time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            # Try adding random elements to the better set
            A_copy = list(best_A)
            B_copy = list(best_B)
            
            # Recompute sums and diffs
            sumsA = set()
            for i, a in enumerate(A_copy):
                for j in range(i, len(A_copy)):
                    sumsA.add(A_copy[i] + A_copy[j])
            sumsB = set()
            for i, b in enumerate(B_copy):
                for j in range(i, len(B_copy)):
                    sumsB.add(B_copy[i] + B_copy[j])
            
            diffsA = set()
            for i, a in enumerate(A_copy):
                for j in range(i + 1, len(A_copy)):
                    diffsA.add(A_copy[j] - A_copy[i])
                    diffsA.add(A_copy[i] - A_copy[j])
            diffsB = set()
            for i, b in enumerate(B_copy):
                for j in range(i + 1, len(B_copy)):
                    diffsB.add(B_copy[j] - B_copy[i])
                    diffsB.add(B_copy[i] - B_copy[j])
            
            # Try adding random elements
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if len(A_copy) >= 64 and len(B_copy) >= 64:
                    break
                
                # Try A
                if len(A_copy) < 64:
                    new_sums = set(x + a for a in A_copy) | {2 * x}
                    if not (new_sums & sumsA):
                        new_diffs = set()
                        for a in A_copy:
                            new_diffs.add(x - a)
                            new_diffs.add(a - x)
                        if not (new_diffs & diffsB):
                            A_copy.append(x)
                            sumsA |= new_sums
                            diffsA |= new_diffs
                            break
                
                # Try B
                if len(B_copy) < 64:
                    new_sums = set(x + b for b in B_copy) | {2 * x}
                    if not (new_sums & sumsB):
                        new_diffs = set()
                        for b in B_copy:
                            new_diffs.add(x - b)
                            new_diffs.add(b - x)
                        if not (new_diffs & diffsA):
                            B_copy.append(x)
                            sumsB |= new_sums
                            diffsB |= new_diffs
                            break
            
            if len(A_copy) + len(B_copy) > best_size:
                if _validate(A_copy, B_copy):
                    best_A, best_B, best_size = A_copy, B_copy, len(A_copy) + len(B_copy)
    
    return [best_A, best_B]
