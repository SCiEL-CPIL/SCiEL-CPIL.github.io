import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 256
    MAX_SIZE = 64
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    def get_diffs(S):
        """Get nonzero difference set of S."""
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                d = abs(S[i] - S[j])
                diffs.add(d)
        return diffs
    
    def get_sums(S):
        """Get pair sums of S (including a+a)."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums
    
    def is_valid_solution(A, B):
        """Full validity check."""
        if not A or not B:
            return False
        if len(A) > MAX_SIZE or len(B) > MAX_SIZE:
            return False
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        # Check disjoint nonzero differences
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True
    
    def can_add_to_set(x, S_set, diffs_S, diffs_other):
        """Check if x can be added to set S without violating constraints."""
        if x in S_set:
            return False
        # Check Sidon condition: for each a in S, |x-a| must not be in diffs_S
        # and must not be in diffs_other, and all new diffs must be distinct
        new_diffs = set()
        for a in S_set:
            d = abs(x - a)
            if d == 0:
                continue
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
            if d in new_diffs:
                return False
            new_diffs.add(d)
        return True
    
    def greedy_build(seed_val):
        """Build a solution using randomized greedy with local search."""
        random.seed(seed_val)
        
        A = []
        B = []
        A_set = set()
        B_set = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidate elements - try different orderings
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # Greedy phase
        for x in candidates:
            valid_A = can_add_to_set(x, A_set, diffs_A, diffs_B)
            valid_B = can_add_to_set(x, B_set, diffs_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    A.append(x)
                    A_set.add(x)
                    for a in A:
                        if a != x:
                            diffs_A.add(abs(x - a))
                elif len(B) < len(A):
                    B.append(x)
                    B_set.add(x)
                    for a in B:
                        if a != x:
                            diffs_B.add(abs(x - a))
                else:
                    if random.random() < 0.5:
                        A.append(x)
                        A_set.add(x)
                        for a in A:
                            if a != x:
                                diffs_A.add(abs(x - a))
                    else:
                        B.append(x)
                        B_set.add(x)
                        for a in B:
                            if a != x:
                                diffs_B.add(abs(x - a))
            elif valid_A:
                A.append(x)
                A_set.add(x)
                for a in A:
                    if a != x:
                        diffs_A.add(abs(x - a))
            elif valid_B:
                B.append(x)
                B_set.add(x)
                for a in B:
                    if a != x:
                        diffs_B.add(abs(x - a))
        
        # Local search: try to improve by removing elements and adding others
        improved = True
        max_iterations = 30
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            for _ in range(20):
                if time.time() >= deadline:
                    break
                
                # Try removing a random element from A and replacing with a better one
                if len(A) > 0:
                    idx = random.randint(0, len(A) - 1)
                    x = A[idx]
                    
                    # Remove x from A
                    A.pop(idx)
                    A_set.discard(x)
                    diffs_A_new = get_diffs(A)
                    
                    # Try to add a different element to A
                    remaining = [y for y in range(1, N + 1) if y not in A_set and y not in B_set]
                    random.shuffle(remaining)
                    
                    added = False
                    for y in remaining[:10]:
                        if can_add_to_set(y, A_set, diffs_A_new, diffs_B):
                            A.append(y)
                            A_set.add(y)
                            diffs_A = diffs_A_new
                            for a in A:
                                if a != y:
                                    diffs_A.add(abs(y - a))
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        # Restore x
                        A.append(x)
                        A_set.add(x)
                        diffs_A = get_diffs(A)
                
                # Try removing a random element from B and replacing with a better one
                if len(B) > 0:
                    idx = random.randint(0, len(B) - 1)
                    x = B[idx]
                    
                    # Remove x from B
                    B.pop(idx)
                    B_set.discard(x)
                    diffs_B_new = get_diffs(B)
                    
                    # Try to add a different element to B
                    remaining = [y for y in range(1, N + 1) if y not in A_set and y not in B_set]
                    random.shuffle(remaining)
                    
                    added = False
                    for y in remaining[:10]:
                        if can_add_to_set(y, B_set, diffs_B_new, diffs_A):
                            B.append(y)
                            B_set.add(y)
                            diffs_B = diffs_B_new
                            for a in B:
                                if a != y:
                                    diffs_B.add(abs(y - a))
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        # Restore x
                        B.append(x)
                        B_set.add(x)
                        diffs_B = get_diffs(B)
        
        # Second greedy pass with remaining elements
        remaining = [x for x in range(1, N + 1) if x not in A_set and x not in B_set]
        random.shuffle(remaining)
        
        for x in remaining:
            valid_A = can_add_to_set(x, A_set, diffs_A, diffs_B)
            valid_B = can_add_to_set(x, B_set, diffs_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    A.append(x)
                    A_set.add(x)
                    for a in A:
                        if a != x:
                            diffs_A.add(abs(x - a))
                elif len(B) < len(A):
                    B.append(x)
                    B_set.add(x)
                    for a in B:
                        if a != x:
                            diffs_B.add(abs(x - a))
                else:
                    if random.random() < 0.5:
                        A.append(x)
                        A_set.add(x)
                        for a in A:
                            if a != x:
                                diffs_A.add(abs(x - a))
                    else:
                        B.append(x)
                        B_set.add(x)
                        for a in B:
                            if a != x:
                                diffs_B.add(abs(x - a))
            elif valid_A:
                A.append(x)
                A_set.add(x)
                for a in A:
                    if a != x:
                        diffs_A.add(abs(x - a))
            elif valid_B:
                B.append(x)
                B_set.add(x)
                for a in B:
                    if a != x:
                        diffs_B.add(abs(x - a))
        
        A.sort()
        B.sort()
        
        # Final validity check
        if is_valid_solution(A, B):
            return A, B
        else:
            return [], []
    
    # Multiple restarts with different seeds
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        seed_val = attempts * 7919 + 12345
        
        A, B = greedy_build(seed_val)
        
        if A and B:
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A.copy()
                best_B = B.copy()
                
                # Early stop if we hit a very high score
                if best_score >= 50:
                    break
    
    # If we didn't find anything, use a simple fallback
    if not best_A or not best_B:
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
