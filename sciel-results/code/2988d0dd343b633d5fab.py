import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    
    # Target lengths
    target_lengths = set(range(3, N + 1))
    
    def covers(S):
        covered = set()
        S_list = sorted(S)
        # s + 1
        for s in S_list:
            covered.add(s + 1)
        # b - a + 2
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                covered.add(S_list[j] - S_list[i] + 2)
        return target_lengths.issubset(covered)
        
    def get_cycles(S):
        cycles = []
        S_list = sorted(S)
        # Map length to cycle
        # We need to return cycles for 3..40 in order
        for L in range(3, N + 1):
            # Check s + 1
            found = False
            for s in S_list:
                if s + 1 == L:
                    cycles.append(list(range(s + 1))) # 0..s
                    found = True
                    break
            if found:
                continue
            
            # Check b - a + 2
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    if S_list[j] - S_list[i] + 2 == L:
                        a = S_list[i]
                        b = S_list[j]
                        # Cycle: 0, a, a+1, ..., b, 0
                        # Vertices: 0, a, a+1, ..., b
                        cyc = [0] + list(range(a, b + 1))
                        cycles.append(cyc)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen if covers(S) is True
                cycles.append([])
        return cycles

    # Search for minimal S
    # Start with small sizes
    for k in range(5, 20):
        # Random search
        trials = 0
        max_trials = 10000
        while trials < max_trials:
            # Random subset of {2..39}
            # Bias towards smaller numbers? Or uniform?
            # Uniform is fine.
            S = set(random.sample(range(2, N), k))
            if covers(S):
                # Found a valid S of size k
                # Construct edges
                edges = []
                for i in range(N - 1):
                    edges.append((i, i + 1))
                for s in S:
                    edges.append((0, s))
                
                # Construct cycles
                cycles = get_cycles(S)
                
                return {
                    'edges': edges,
                    'cycles': cycles
                }
            trials += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Fallback: use a known good construction or the seed
    # Seed: S = {2..39} (size 38)
    S = set(range(2, N))
    edges = []
    for i in range(N - 1):
        edges.append((i, i + 1))
    for s in S:
        edges.append((0, s))
    cycles = get_cycles(S)
    return {
        'edges': edges,
        'cycles': cycles
    }
