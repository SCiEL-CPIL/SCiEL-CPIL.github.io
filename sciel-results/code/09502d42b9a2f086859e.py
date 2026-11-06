"""
Parallel Tempering (Replica Exchange) for 4-AP-free set maximizing 3-AP count.

Uses incremental delta calculations for 3-AP and 4-AP counts to allow
high-throughput move evaluation.
"""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # Safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute APs
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    three_ap_indices = [[] for _ in range(N)] # For each element, list of 3-AP indices it's in
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            idx = len(three_aps)
            ap = (a, a + d, a + 2 * d)
            three_aps.append(ap)
            for x in ap:
                three_ap_indices[x].append(idx)
                
    # 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    four_ap_indices = [[] for _ in range(N)] # For each element, list of 4-AP indices it's in
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            idx = len(four_aps)
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            four_aps.append(ap)
            for x in ap:
                four_ap_indices[x].append(idx)
                
    num_3aps = len(three_aps)
    num_4aps = len(four_aps)
    
    # State representation for a replica
    # A: list of bools (0 or 1)
    # n3: count of 3-APs fully contained in A
    # n4: count of 4-APs fully contained in A
    
    def calculate_initial_state(A):
        """Calculate n3 and n4 for a given set A."""
        n3 = 0
        for ap in three_aps:
            if A[ap[0]] and A[ap[1]] and A[ap[2]]:
                n3 += 1
        n4 = 0
        for ap in four_aps:
            if A[ap[0]] and A[ap[1]] and A[ap[2]] and A[ap[3]]:
                n4 += 1
        return n3, n4

    def delta_3ap(A, x):
        """
        Calculate change in number of 3-APs if we toggle x.
        If x is 0 (adding): Count 3-APs containing x where the other 2 are 1.
        If x is 1 (removing): Count 3-APs containing x where the other 2 are 1.
        """
        delta = 0
        for ap_idx in three_ap_indices[x]:
            a, b, c = three_aps[ap_idx]
            # The other two elements
            if x == a:
                o1, o2 = b, c
            elif x == b:
                o1, o2 = a, c
            else:
                o1, o2 = a, b
            
            if A[o1] and A[o2]:
                # If adding x, this AP becomes complete (+1)
                # If removing x, this AP becomes incomplete (-1)
                if A[x]:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def delta_4ap(A, x):
        """
        Calculate change in number of 4-APs if we toggle x.
        """
        delta = 0
        for ap_idx in four_ap_indices[x]:
            a, b, c, d = four_aps[ap_idx]
            # The other three elements
            if x == a:
                others = [b, c, d]
            elif x == b:
                others = [a, c, d]
            elif x == c:
                others = [a, b, d]
            else:
                others = [a, b, c]
            
            if all(A[ox] for ox in others):
                if A[x]:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Parallel Tempering Setup
    # Define temperatures
    # We want to maximize Score = n3 - P * n4
    # Let's use P = 1000 to strongly penalize 4-APs.
    P = 1000
    
    # Number of replicas
    num_replicas = 16
    # Geometric distribution of temperatures
    # T_high allows accepting bad moves (high n4, low n3)
    # T_low focuses on optimization
    T_high = 50.0 
    T_low = 0.1
    temps = []
    for i in range(num_replicas):
        # Linear interpolation in log space
        log_t = math.log(T_high) + (math.log(T_low) - math.log(T_high)) * (i / (num_replicas - 1))
        temps.append(math.exp(log_t))
        
    # Initialize replicas
    # Replica 0 (coldest) starts with a greedy construction
    # Others start with random or slightly perturbed sets
    
    def greedy_construct():
        """Greedy: add elements that maximize 3-APs without creating 4-APs."""
        A = [0] * N
        # We prioritize elements with high potential 3-APs
        # Simple heuristic: iterate through elements, add if no 4-AP created
        # To make it better, we can sort by "degree" in 3-AP graph
        weights = [len(three_ap_indices[i]) for i in range(N)]
        order = sorted(range(N), key=lambda x: -weights[x])
        
        for x in order:
            # Check if adding x creates a 4-AP
            creates_4ap = False
            for ap_idx in four_ap_indices[x]:
                a, b, c, d = four_aps[ap_idx]
                if x == a:
                    others = [b, c, d]
                elif x == b:
                    others = [a, c, d]
                elif x == c:
                    others = [a, b, d]
                else:
                    others = [a, b, c]
                
                if all(A[ox] for ox in others):
                    creates_4ap = True
                    break
            
            if not creates_4ap:
                A[x] = 1
                
        return A

    # Create initial states
    replicas = []
    for i in range(num_replicas):
        if i == 0:
            A = greedy_construct()
        else:
            # Random initialization with some density
            # Start with a random subset of size ~60
            A = [0] * N
            # Pick random elements
            # To avoid obvious 4-APs, we can just start random and let PT fix it,
            # but starting closer to feasible is better.
            # Let's use a simple random set
            for j in range(N):
                if rng.random() < 0.25:
                    A[j] = 1
            # Optional: quick repair to remove 4-APs
            # We'll rely on the high temp replicas to escape, but initial n4 should be low
            # Let's do a few rounds of removing elements from 4-APs
            for _ in range(50):
                found = False
                for ap_idx in range(num_4aps):
                    a, b, c, d = four_aps[ap_idx]
                    if A[a] and A[b] and A[c] and A[d]:
                        # Remove one randomly
                        to_remove = rng.choice([a, b, c, d])
                        A[to_remove] = 0
                        found = True
                        break
                if not found:
                    break
        
        n3, n4 = calculate_initial_state(A)
        replicas.append({
            'A': A,
            'n3': n3,
            'n4': n4,
            'T': temps[i]
        })
    
    # Track best feasible solution
    best_n3 = -1
    best_A = None
    
    # Check initial states
    for rep in replicas:
        if rep['n4'] == 0:
            if rep['n3'] > best_n3:
                best_n3 = rep['n3']
                best_A = rep['A'].copy()
    
    # Main Loop
    # Parameters
    swap_interval = 10 # Swap every 10 iterations per replica
    max_iters = 100000 # Safety limit on iterations
    
    iters = 0
    while iters < max_iters:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Perform one move for each replica
        for i in range(num_replicas):
            rep = replicas[i]
            A = rep['A']
            n3 = rep['n3']
            n4 = rep['n4']
            T = rep['T']
            
            # Pick a random element to toggle
            x = rng.randint(0, N - 1)
            
            d3 = delta_3ap(A, x)
            d4 = delta_4ap(A, x)
            
            # Score change: Delta E = -(Delta Score)
            # Score = n3 - P*n4
            # Delta Score = d3 - P*d4
            # Delta E = - (d3 - P*d4) = P*d4 - d3
            
            delta_E = P * d4 - d3
            
            # Metropolis Acceptance
            if delta_E <= 0:
                accept = True
            else:
                # exp(-delta_E / T)
                # If T is small, this is small.
                # If T is large, this is ~1.
                # We must be careful with overflow/underflow
                if T == 0:
                    accept = False
                else:
                    ratio = math.exp(-delta_E / T)
                    accept = rng.random() < ratio
            
            if accept:
                # Update state
                A[x] = 1 - A[x]
                n3 += d3
                n4 += d4
                rep['n3'] = n3
                rep['n4'] = n4
                
                # Update best
                if n4 == 0 and n3 > best_n3:
                    best_n3 = n3
                    best_A = A.copy()
            
            iters += 1
            if iters % swap_interval == 0:
                break # Break out of replica loop to do swaps
                
        # Perform Replica Exchange
        # We swap replicas i and i+1 if it improves the total energy
        # Or simply try to swap adjacent replicas
        for i in range(num_replicas - 1):
            rep1 = replicas[i]
            rep2 = replicas[i+1]
            
            T1 = rep1['T']
            T2 = rep2['T']
            
            E1 = -(rep1['n3'] - P * rep1['n4']) # Energy = -Score
            E2 = -(rep2['n3'] - P * rep2['n4'])
            
            # If we swap, new energies are E2 at T1, E1 at T2
            # Delta E_swap = (E2/T1 + E1/T2) - (E1/T1 + E2/T2)
            # = E2(1/T1 - 1/T2) + E1(1/T2 - 1/T1)
            # = (E2 - E1) * (1/T1 - 1/T2)
            
            if T1 == T2:
                continue
                
            delta_swap = (E2 - E1) * (1.0/T1 - 1.0/T2)
            
            if delta_swap <= 0:
                accept_swap = True
            else:
                # Use average temperature or T1? Standard is usually T1 or T2.
                # Let's use T1.
                accept_swap = rng.random() < math.exp(-delta_swap / T1)
            
            if accept_swap:
                # Swap the states (A, n3, n4) but keep temperatures fixed to positions
                # Actually, in PT, we swap the *configurations* between the temperature levels.
                # So replicas[i] gets rep2's config, replicas[i+1] gets rep1's config.
                rep1['A'], rep2['A'] = rep2['A'], rep1['A']
                rep1['n3'], rep2['n3'] = rep2['n3'], rep1['n3']
                rep1['n4'], rep2['n4'] = rep2['n4'], rep1['n4']
                
                # Update best if the swapped-in config is better
                if rep1['n4'] == 0 and rep1['n3'] > best_n3:
                    best_n3 = rep1['n3']
                    best_A = rep1['A'].copy()
                if rep2['n4'] == 0 and rep2['n3'] > best_n3:
                    best_n3 = rep2['n3']
                    best_A = rep2['A'].copy()

    # Final Result
    if best_A is None:
        # Fallback: return a single element
        return [1]
        
    result = [x + 1 for x in range(N) if best_A[x]]
    return result

# Run construct to verify it works quickly in isolation if needed
# But the evaluator calls construct().

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Result size: {len(res)}")
    # Verify no 4-APs
    # (Verification logic omitted for brevity in main block, assumed correct in construct)
