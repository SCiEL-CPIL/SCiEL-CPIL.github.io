import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    
    # Use a list of booleans or a set for visited. 
    # Since the number of subsets is 2^12 = 4096, we can map frozensets to integers for speed.
    # But for simplicity and correctness first, let's use a set of frozensets.
    # Optimization: Precompute transitions? No, a_trans and b_trans change.
    
    visited = {start}
    q = deque([start])
    dist = 0
    
    while q:
        size = len(q)
        for _ in range(size):
            curr = q.popleft()
            if len(curr) == 1:
                return dist
            
            # Apply 'a'
            next_a = frozenset(a_trans[i] for i in curr)
            if next_a not in visited:
                visited.add(next_a)
                q.append(next_a)
            
            # Apply 'b'
            next_b = frozenset(b_trans[i] for i in curr)
            if next_b not in visited:
                visited.add(next_b)
                q.append(next_b)
        
        dist += 1
        
        # Safety check: if we've visited all possible subsets, it's not synchronizing
        if len(visited) == (1 << N):
            break
            
    return -1

def construct():
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin for overhead/return
    
    # Initial best: The "cycle + sink" automaton
    # a: i -> (i+1)%N
    # b: i -> 0
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_score = calculate_threshold(best_a, best_b)
    
    # Store best solution
    best_a_copy = best_a[:]
    best_b_copy = best_b[:]
    
    # Local search strategy:
    # 1. Randomly pick a letter (a or b) and a state.
    # 2. Randomly change its destination.
    # 3. Evaluate. If better, keep. If worse, revert.
    # 4. If stuck, do a "kick" (multiple random changes) to escape local optima.
    
    no_improvement = 0
    max_no_improvement = 2000
    kick_interval = 500
    
    while time.time() - start_time < timeout:
        # Check if we've hit the theoretical max for N=12?
        # Černý's conjecture says N-1. But for small N, sometimes it's larger?
        # Actually, for N=12, the known maximum is likely around 11 or slightly more?
        # Wait, the seed score was 9.0.
        # If best_score == N-1 (11), we might stop?
        # But the task says "maximizing". Let's not assume N-1 is max for all N.
        # For N=3, max is 2. For N=4, max is 3? 
        # Actually, the maximum reset length for N states is known to be N-1 for N<=? 
        # No, for N=4, max is 3. For N=5, max is 4?
        # Let's just search.
        
        # Random mutation
        letter = random.choice([0, 1])
        state = random.randint(0, N - 1)
        
        if letter == 0:
            old_val = best_a[state]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            best_a[state] = new_val
            new_score = calculate_threshold(best_a, best_b)
            if new_score > best_score:
                best_score = new_score
                best_a_copy = best_a[:]
                best_b_copy = best_b[:]
                no_improvement = 0
            else:
                best_a[state] = old_val
                no_improvement += 1
        else:
            old_val = best_b[state]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            best_b[state] = new_val
            new_score = calculate_threshold(best_a, best_b)
            if new_score > best_score:
                best_score = new_score
                best_a_copy = best_a[:]
                best_b_copy = best_b[:]
                no_improvement = 0
            else:
                best_b[state] = old_val
                no_improvement += 1
        
        # Kick strategy: If stuck for a while, make several random changes
        if no_improvement > max_no_improvement:
            # Make 5 random changes
            for _ in range(5):
                l = random.choice([0, 1])
                s = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if l == 0:
                    best_a[s] = v
                else:
                    best_b[s] = v
            
            kick_score = calculate_threshold(best_a, best_b)
            if kick_score > best_score:
                best_score = kick_score
                best_a_copy = best_a[:]
                best_b_copy = best_b[:]
                no_improvement = 0
            else:
                # Revert to best known
                best_a = best_a_copy[:]
                best_b = best_b_copy[:]
                no_improvement = 0 # Reset counter to try again from best
        
    return [best_a_copy, best_b_copy]
