import math
import random
import time

def construct():
    N = 512
    # Precompute LCM compatibility
    # compat[i] is a set of j such that lcm(i, j) <= N
    # To save memory/time, we can just compute on the fly or use a bitset.
    # 512 is small, so a list of sets is fine.
    compat = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if math.lcm(i, j) <= N:
                compat[i].add(j)
                if i != j:
                    compat[j].add(i)
    
    # Greedy initialization
    S = set()
    # Sort candidates by value, maybe prioritize small ones
    for i in range(1, N + 1):
        # Check if i is compatible with all in S
        if all(i in compat[j] for j in S):
            S.add(i)
            
    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    deadline = 60.0 # Safety margin
    
    # Local Search
    # We try to improve S
    # Moves:
    # 1. Add a node
    # 2. Swap a node
    
    while time.time() < deadline:
        improved = False
        
        # Try to add
        # Randomly sample candidates not in S
        candidates = [i for i in range(1, N + 1) if i not in S]
        random.shuffle(candidates)
        
        for v in candidates:
            # Check if v is compatible with all in S
            if all(v in compat[u] for u in S):
                S.add(v)
                improved = True
                break # Restart loop to pick from new candidates
        
        if improved:
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        
        # Try to swap
        # Pick random u in S, v not in S
        # We want to find a swap that doesn't reduce size, but might enable future additions
        # Actually, simple swap keeps size same.
        # We should try swaps that are "neutral" but might break a bottleneck.
        
        # To be more effective, we can try to remove a node that has many conflicts with outside nodes?
        # Or just random swap.
        
        # Let's try a few random swaps per iteration
        for _ in range(10):
            u = random.choice(list(S))
            v = random.choice(candidates)
            
            # Check if S - {u} + {v} is valid
            # v must be compatible with all in S - {u}
            # i.e., v is compatible with all w in S where w != u
            valid_swap = True
            for w in S:
                if w == u:
                    continue
                if v not in compat[w]:
                    valid_swap = False
                    break
            
            if valid_swap:
                S.remove(u)
                S.add(v)
                # If size is same, we don't update best yet, but we changed state
                # We might want to track best size
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                # Accept the swap and break to restart add-phase
                improved = True
                break
        
        if not improved:
            # If we can't add or swap, we might be stuck in a local optimum.
            # We can try a 2-for-2 swap or just restart?
            # Given the time, let's just keep trying random swaps.
            # Or, we can try to remove a node and see if we can add two?
            # That's complex.
            # Let's just continue the loop.
            pass
            
    return list(best_S)
