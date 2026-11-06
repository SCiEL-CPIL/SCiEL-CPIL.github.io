# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    n = 256
    max_elements = 64
    
    def time_left():
        return deadline - time.time()
    
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
        """Get nonzero difference set of aset."""
        diffs = set()
        for i in range(len(aset)):
            for j in range(len(aset)):
                if i != j:
                    diffs.add(aset[j] - aset[i])
        return diffs
    
    def get_diffs_abs(aset):
        """Get absolute nonzero difference set."""
        diffs = set()
        for i in range(len(aset)):
            for j in range(i+1, len(aset)):
                d = abs(aset[j] - aset[i])
                if d != 0:
                    diffs.add(d)
        return diffs
    
    def can_add_to_sidon(x, aset):
        """Check if x can be added to aset while maintaining Sidon property."""
        for a in aset:
            s = x + a
            if s in aset_sums_cache.get(id(aset), set()):
                # Need to check more carefully
                pass
        # Direct check
        for a in aset:
            for b in aset:
                if a <= b:
                    if x + a == a + b and x != b:
                        return False
                    if x + a == b + a and x != b:
                        return False
        # Better: check all new sums against existing sums
        existing_sums = set()
        for i in range(len(aset)):
            for j in range(i, len(aset)):
                existing_sums.add(aset[i] + aset[j])
        for a in aset:
            s = x + a
            if s in existing_sums:
                return False
        s2 = x + x
        if s2 in existing_sums:
            return False
        return True
    
    def can_add_diff_disjoint(x, aset, other_diffs):
        """Check if adding x to aset keeps difference sets disjoint from other_diffs."""
        for a in aset:
            d = x - a
            if d != 0 and d in other_diffs:
                return False
            d = a - x
            if d != 0 and d in other_diffs:
                return False
        return True
    
    def greedy_build(order_a, order_b, max_iter=None):
        """Build A and B greedily following given orders."""
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        iter_count = 0
        for x in order_a:
            if time_left() < 2.0:
                break
            if len(A) >= max_elements:
                break
            iter_count += 1
            if max_iter and iter_count >= max_iter:
                break
            
            # Check Sidon for A
            ok = True
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
            if ok:
                s = x + x
                if s in sums_A:
                    ok = False
            if not ok:
                continue
            
            # Check difference disjointness
            ok = True
            for a in A:
                d1 = x - a
                d2 = a - x
                if d1 != 0 and d1 in diffs_B:
                    ok = False
                    break
                if d2 != 0 and d2 in diffs_B:
                    ok = False
                    break
            if not ok:
                continue
            
            # Add x to A
            A.append(x)
            for a in A[:-1]:
                sums_A.add(x + a)
            sums_A.add(x + x)
            for a in A[:-1]:
                diffs_A.add(x - a)
                diffs_A.add(a - x)
        
        for x in order_b:
            if time_left() < 2.0:
                break
            if len(B) >= max_elements:
                break
            iter_count += 1
            if max_iter and iter_count >= max_iter:
                break
            
            # Check Sidon for B
            ok = True
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
            if ok:
                s = x + x
                if s in sums_B:
                    ok = False
            if not ok:
                continue
            
            # Check difference disjointness
            ok = True
            for b in B:
                d1 = x - b
                d2 = b - x
                if d1 != 0 and d1 in diffs_A:
                    ok = False
                    break
                if d2 != 0 and d2 in diffs_A:
                    ok = False
                    break
            if not ok:
                continue
            
            # Add x to B
            B.append(x)
            for b in B[:-1]:
                sums_B.add(x + b)
            sums_B.add(x + x)
            for b in B[:-1]:
                diffs_B.add(x - b)
                diffs_B.add(b - x)
        
        return A, B
    
    def local_improve(A, B, max_rounds=100):
        """Try to improve A and B by removing elements and re-adding better ones."""
        A = list(A)
        B = list(B)
        
        for _ in range(max_rounds):
            if time_left() < 3.0:
                break
            
            improved = False
            
            # Try to add more elements to A
            for x in range(1, n+1):
                if time_left() < 3.0:
                    break
                if x in A:
                    continue
                if len(A) >= max_elements:
                    break
                
                # Check Sidon for A
                ok = True
                existing_sums = set()
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        existing_sums.add(A[i] + A[j])
                
                for a in A:
                    s = x + a
                    if s in existing_sums:
                        ok = False
                        break
                if ok:
                    s = x + x
                    if s in existing_sums:
                        ok = False
                
                if not ok:
                    continue
                
                # Check difference disjointness
                diffs_B = get_diffs(B)
                ok = True
                for a in A:
                    d1 = x - a
                    d2 = a - x
                    if d1 != 0 and d1 in diffs_B:
                        ok = False
                        break
                    if d2 != 0 and d2 in diffs_B:
                        ok = False
                        break
                if not ok:
                    continue
                
                A.append(x)
                improved = True
                break
            
            # Try to add more elements to B
            for x in range(1, n+1):
                if time_left() < 3.0:
                    break
                if x in B:
                    continue
                if len(B) >= max_elements:
                    break
                
                # Check Sidon for B
                ok = True
                existing_sums = set()
                for i in range(len(B)):
                    for j in range(i, len(B)):
                        existing_sums.add(B[i] + B[j])
                
                for b in B:
                    s = x + b
                    if s in existing_sums:
                        ok = False
                        break
                if ok:
                    s = x + x
                    if s in existing_sums:
                        ok = False
                
                if not ok:
                    continue
                
                # Check difference disjointness
                diffs_A = get_diffs(A)
                ok = True
                for b in B:
                    d1 = x - b
                    d2 = b - x
                    if d1 != 0 and d1 in diffs_A:
                        ok = False
                        break
                    if d2 != 0 and d2 in diffs_A:
                        ok = False
                        break
                if not ok:
                    continue
                
                B.append(x)
                improved = True
                break
            
            if not improved:
                break
        
        return A, B
    
    def remove_and_replace(A, B, max_rounds=50):
        """Try removing one element and replacing with a better one."""
        A = list(A)
        B = list(B)
        
        for _ in range(max_rounds):
            if time_left() < 4.0:
                break
            
            improved = False
            best_gain = 0
            best_action = None
            
            # Try removing each element from A and see if we can add more
            for i in range(len(A)):
                if time_left() < 4.0:
                    break
                
                x_remove = A[i]
                A_temp = A[:i] + A[i+1:]
                
                # Try to add new elements to A_temp
                added_count = 0
                for x in range(1, n+1):
                    if x in A_temp or x == x_remove:
                        continue
                    if len(A_temp) + added_count >= max_elements:
                        break
                    
                    # Check Sidon for A_temp
                    ok = True
                    existing_sums = set()
                    for a1 in A_temp:
                        for a2 in A_temp:
                            if a1 <= a2:
                                existing_sums.add(a1 + a2)
                    
                    for a in A_temp:
                        s = x + a
                        if s in existing_sums:
                            ok = False
                            break
                    if ok:
                        s = x + x
                        if s in existing_sums:
                            ok = False
                    
                    if not ok:
                        continue
                    
                    # Check difference disjointness
                    diffs_B = get_diffs(B)
                    ok = True
                    for a in A_temp:
                        d1 = x - a
                        d2 = a - x
                        if d1 != 0 and d1 in diffs_B:
                            ok = False
                            break
                        if d2 != 0 and d2 in diffs_B:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    A_temp.append(x)
                    added_count += 1
                    if added_count >= 2:  # net gain of at least 1
                        break
                
                if added_count > 1:  # net gain
                    gain = added_count - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ('A', i, A_temp)
            
            # Try removing each element from B and see if we can add more
            for i in range(len(B)):
                if time_left() < 4.0:
                    break
                
                x_remove = B[i]
                B_temp = B[:i] + B[i+1:]
                
                # Try to add new elements to B_temp
                added_count = 0
                for x in range(1, n+1):
                    if x in B_temp or x == x_remove:
                        continue
                    if len(B_temp) + added_count >= max_elements:
                        break
                    
                    # Check Sidon for B_temp
                    ok = True
                    existing_sums = set()
                    for b1 in B_temp:
                        for b2 in B_temp:
                            if b1 <= b2:
                                existing_sums.add(b1 + b2)
                    
                    for b in B_temp:
                        s = x + b
                        if s in existing_sums:
                            ok = False
                            break
                    if ok:
                        s = x + x
                        if s in existing_sums:
                            ok = False
                    
                    if not ok:
                        continue
                    
                    # Check difference disjointness
                    diffs_A = get_diffs(A)
                    ok = True
                    for b in B_temp:
                        d1 = x - b
                        d2 = b - x
                        if d1 != 0 and d1 in diffs_A:
                            ok = False
                            break
                        if d2 != 0 and d2 in diffs_A:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    B_temp.append(x)
                    added_count += 1
                    if added_count >= 2:
                        break
                
                if added_count > 1:
                    gain = added_count - 1
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ('B', i, B_temp)
            
            if best_action is not None:
                side, idx, new_set = best_action
                if side == 'A':
                    A = new_set
                else:
                    B = new_set
                improved = True
            else:
                break
        
        return A, B
    
    # Initial greedy build
    base_order = list(range(1, n+1))
    
    best_A, best_B = [], []
    best_size = 0
    
    # Try multiple random orderings
    num_trials = 30
    for trial in range(num_trials):
        if time_left() < 10.0:
            break
        
        if trial == 0:
            order_a = base_order[:]
            order_b = base_order[:]
        else:
            order_a = base_order[:]
            order_b = base_order[:]
            random.shuffle(order_a)
            random.shuffle(order_b)
        
        A, B = greedy_build(order_a, order_b)
        
        # Local improvement
        A, B = local_improve(A, B, max_rounds=50)
        
        # Remove and replace
        A, B = remove_and_replace(A, B, max_rounds=20)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
    
    # Final polish: try to add any remaining elements
    for x in range(1, n+1):
        if time_left() < 2.0:
            break
        if x in best_A:
            continue
        if len(best_A) >= max_elements:
            continue
        
        # Check Sidon for best_A
        ok = True
        existing_sums = set()
        for i in range(len(best_A)):
            for j in range(i, len(best_A)):
                existing_sums.add(best_A[i] + best_A[j])
        
        for a in best_A:
            s = x + a
            if s in existing_sums:
                ok = False
                break
        if ok:
            s = x + x
            if s in existing_sums:
                ok = False
        
        if not ok:
            continue
        
        # Check difference disjointness
        diffs_B = get_diffs(best_B)
        ok = True
        for a in best_A:
            d1 = x - a
            d2 = a - x
            if d1 != 0 and d1 in diffs_B:
                ok = False
                break
            if d2 != 0 and d2 in diffs_B:
                ok = False
                break
        if not ok:
            continue
        
        best_A.append(x)
    
    for x in range(1, n+1):
        if time_left() < 2.0:
            break
        if x in best_B:
            continue
        if len(best_B) >= max_elements:
            continue
        
        # Check Sidon for best_B
        ok = True
        existing_sums = set()
        for i in range(len(best_B)):
            for j in range(i, len(best_B)):
                existing_sums.add(best_B[i] + best_B[j])
        
        for b in best_B:
            s = x + b
            if s in existing_sums:
                ok = False
                break
        if ok:
            s = x + x
            if s in existing_sums:
                ok = False
        
        if not ok:
            continue
        
        # Check difference disjointness
        diffs_A = get_diffs(best_A)
        ok = True
        for b in best_B:
            d1 = x - b
            d2 = b - x
            if d1 != 0 and d1 in diffs_A:
                ok = False
                break
            if d2 != 0 and d2 in diffs_A:
                ok = False
                break
        if not ok:
            continue
        
        best_B.append(x)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
