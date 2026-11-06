"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    n = 256
    max_size_each = 64
    
    best_A = []
    best_B = []
    best_total = 0
    
    def check_sidon(aset):
        """Check if aset is a Sidon set (unique pair sums including a+a)."""
        sums = set()
        for i in range(len(aset)):
            for j in range(i, len(aset)):
                s = aset[i] + aset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(aset):
        """Get all nonzero differences from aset."""
        diffs = set()
        for i in range(len(aset)):
            for j in range(len(aset)):
                if i != j:
                    d = aset[i] - aset[j]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def can_add_to_sidon(x, aset, used_sums):
        """Check if x can be added to aset while maintaining Sidon property."""
        for a in aset:
            s = x + a
            if s in used_sums:
                return False
            used_sums.add(s)
        # Check x + x
        if x + x in used_sums:
            # Need to undo the additions
            for a in aset:
                used_sums.discard(x + a)
            return False
        used_sums.add(x + x)
        # All good, but we need to undo for the caller
        for a in aset:
            used_sums.discard(x + a)
        used_sums.discard(x + x)
        return True
    
    def get_used_sums(aset):
        """Get the set of all pair sums from aset."""
        sums = set()
        for i in range(len(ASET)):
            for j in range(i, len(aset)):
                sums.add(aset[i] + aset[j])
        return sums
    
    def greedy_construct(order_A, order_B):
        """Greedy construction with given orders."""
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order_A:
            if x > n:
                continue
            # Check Sidon for A
            ok = True
            new_sums = set()
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
                new_sums.add(s)
            if ok and x + x not in sums_A:
                new_sums.add(x + x)
            else:
                ok = False
            
            if not ok:
                continue
            
            # Check differences don't conflict with B
            new_diffs = set()
            for a in A:
                d = x - a
                if d != 0:
                    new_diffs.add(d)
                d2 = a - x
                if d2 != 0:
                    new_diffs.add(d2)
            
            if new_diffs & diffs_B:
                continue
            
            # Add x to A
            A.append(x)
            sums_A.update(new_sums)
            diffs_A.update(new_diffs)
        
        for x in order_B:
            if x > n:
                continue
            # Check Sidon for B
            ok = True
            new_sums = set()
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
                new_sums.add(s)
            if ok and x + x not in sums_B:
                new_sums.add(x + x)
            else:
                ok = False
            
            if not ok:
                continue
            
            # Check differences don't conflict with A
            new_diffs = set()
            for b in B:
                d = x - b
                if d != 0:
                    new_diffs.add(d)
                d2 = b - x
                if d2 != 0:
                    new_diffs.add(d2)
            
            if new_diffs & diffs_A:
                continue
            
            # Add x to B
            B.append(x)
            sums_B.update(new_sums)
            diffs_B.update(new_diffs)
        
        return A, B
    
    # Try multiple random orders
    random.seed(42)
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Create random orders
        order_A = list(range(1, n + 1))
        order_B = list(range(1, n + 1))
        random.shuffle(order_A)
        random.shuffle(order_B)
        
        A, B = greedy_construct(order_A, order_B)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Also try a deterministic approach: alternate between A and B
    if time.time() - start_time < time_limit:
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in range(1, n + 1):
            # Try adding to A
            ok_A = True
            new_sums_A = set()
            for a in A:
                s = x + a
                if s in sums_A:
                    ok_A = False
                    break
                new_sums_A.add(s)
            if ok_A and x + x not in sums_A:
                new_sums_A.add(x + x)
            else:
                ok_A = False
            
            if ok_A:
                new_diffs_A = set()
                for a in A:
                    d = x - a
                    if d != 0:
                        new_diffs_A.add(d)
                    d2 = a - x
                    if d2 != 0:
                        new_diffs_A.add(d2)
                
                if not (new_diffs_A & diffs_B):
                    A.append(x)
                    sums_A.update(new_sums_A)
                    diffs_A.update(new_diffs_A)
                    continue
            
            # Try adding to B
            ok_B = True
            new_sums_B = set()
            for b in B:
                s = x + b
                if s in sums_B:
                    ok_B = False
                    break
                new_sums_B.add(s)
            if ok_B and x + x not in sums_B:
                new_sums_B.add(x + x)
            else:
                ok_B = False
            
            if ok_B:
                new_diffs_B = set()
                for b in B:
                    d = x - b
                    if d != 0:
                        new_diffs_B.add(d)
                    d2 = b - x
                    if d2 != 0:
                        new_diffs_B.add(d2)
                
                if not (new_diffs_B & diffs_A):
                    B.append(x)
                    sums_B.update(new_sums_B)
                    diffs_B.update(new_diffs_B)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Verify the best solution
    if best_A and best_B:
        # Check Sidon for A
        sums = set()
        for i in range(len(best_A)):
            for j in range(i, len(best_A)):
                s = best_A[i] + best_A[j]
                if s in sums:
                    # Invalid, return empty
                    return [[1], [2]]
                sums.add(s)
        
        # Check Sidon for B
        sums = set()
        for i in range(len(best_B)):
            for j in range(i, len(best_B)):
                s = best_B[i] + best_B[j]
                if s in sums:
                    return [[1], [2]]
                sums.add(s)
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(best_A)):
            for j in range(len(best_A)):
                if i != j:
                    d = best_A[i] - best_A[j]
                    if d != 0:
                        diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(best_B)):
            for j in range(len(best_B)):
                if i != j:
                    d = best_B[i] - best_B[j]
                    if d != 0:
                        diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return [[1], [2]]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
