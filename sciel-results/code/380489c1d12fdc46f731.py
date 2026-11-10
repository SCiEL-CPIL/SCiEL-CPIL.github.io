import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    best_total = 0
    best_A = []
    best_B = []
    
    def _check_sidon_and_disjoint(A, B, a, b):
        """Check if adding a to A and b to B maintains Sidon property and disjoint differences."""
        # Check Sidon property for A
        if a in A:
            return False
        new_A = list(A) + [a]
        sums_A = [x + y for i, x in enumerate(new_A) for y in new_A[i:]]
        if len(sums_A) != len(set(sums_A)):
            return False
        
        # Check Sidon property for B
        if b in B:
            return False
        new_B = list(B) + [b]
        sums_B = [x + y for i, x in enumerate(new_B) for y in new_B[i:]]
        if len(sums_B) != len(set(sums_B)):
            return False
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(new_A)):
            for j in range(i + 1, len(new_A)):
                diff = abs(new_A[i] - new_A[j])
                diffs_A.add(diff)
        
        diffs_B = set()
        for i in range(len(new_B)):
            for j in range(i + 1, len(new_B)):
                diff = abs(new_B[i] - new_B[j])
                diffs_B.add(diff)
        
        if diffs_A & diffs_B:
            return False
        
        return True
    
    def _greedy_random(seed, max_time):
        random.seed(seed)
        A = []
        B = []
        used = set()
        
        # Start with random initial elements
        if random.random() < 0.5:
            A.append(1)
            used.add(1)
        else:
            B.append(1)
            used.add(1)
        
        while time.time() < max_time:
            # Try to add to the smaller set
            if len(A) <= len(B):
                side = 'A'
            else:
                side = 'B'
            
            # Try random candidates
            candidates = [x for x in range(1, 257) if x not in used]
            random.shuffle(candidates)
            
            added = False
            for c in candidates:
                if side == 'A':
                    if _check_sidon_and_disjoint(A, B, c, -1):
                        A.append(c)
                        used.add(c)
                        added = True
                        break
                else:
                    if _check_sidon_and_disjoint(A, B, -1, c):
                        B.append(c)
                        used.add(c)
                        added = True
                        break
            
            if not added:
                # Try the other side
                if side == 'A':
                    side = 'B'
                else:
                    side = 'A'
                
                for c in candidates:
                    if side == 'A':
                        if _check_sidon_and_disjoint(A, B, c, -1):
                            A.append(c)
                            used.add(c)
                            added = True
                            break
                    else:
                        if _check_sidon_and_disjoint(A, B, -1, c):
                            B.append(c)
                            used.add(c)
                            added = True
                            break
                
                if not added:
                    break
        
        return A, B
    
    # Multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() >= deadline:
            break
        
        A, B = _greedy_random(i, deadline)
        total = len(A) + len(B)
        
        if total > best_total:
            best_total = total
            best_A = A
            best_B = B
    
    return [best_A, best_B]
