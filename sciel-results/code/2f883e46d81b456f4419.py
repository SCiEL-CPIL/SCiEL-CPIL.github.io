import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (evaluator kills at 90s)

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    # Precompute the initial state masks for all 2^N inputs
    # state[j] is a bitmask of length 2^N, where bit k is 1 if input k has a 1 on wire j
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            # Optimization: if state[i] is subset of state[j], no change needed?
            # Actually, min goes to i, max to j.
            # new_i = state[i] & state[j]
            # new_j = state[i] | state[j]
            # If state[i] == state[j], no change.
            # If state[i] & ~state[j] == 0 and state[j] & ~state[i] == 0, no change.
            # Let's just compute.
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Check if sorted: state[k] should be a subset of state[k+1]
        # i.e., state[k] & ~state[k+1] should be 0
        # Wait, if sorted ascending, wire 0 has smallest values.
        # For boolean inputs, 0 < 1.
        # So if wire 0 has a 1, all subsequent wires must have 1? No.
        # If wire 0 is 1, it means for some input, wire 0 was 1.
        # In a sorted network, if wire 0 is 1, then wire 1 must be 1?
        # Yes, because 1 > 0. If wire 0 has a 1, it means that input had at least one 1 at wire 0.
        # Actually, the condition for sorting boolean inputs is:
        # For all k < m, if state[k] has a 1, then state[m] must have a 1 for that same input?
        # No. The condition is that the vector of bits for each input is sorted.
        # Input k is a vector v_k of length N.
        # v_k must be sorted: v_k[0] <= v_k[1] <= ... <= v_k[N-1].
        # This is equivalent to: if v_k[j] == 1 and v_k[i] == 0 for i < j, it's unsorted.
        # In our bitmask representation:
        # state[j] has bit k set if v_k[j] == 1.
        # Condition: If v_k[j] == 1 and v_k[i] == 0, then invalid.
        # This means: for any i < j, if state[j] has bit k set, state[i] must have bit k set?
        # No. If v_k[j]=1 and v_k[i]=0, then state[j] has bit k, state[i] does not.
        # So we need: state[j] & ~state[i] == 0 for all i < j?
        # Let's check: if state[j] has bit k, and state[i] does not, then v_k[j]=1, v_k[i]=0.
        # This violates sorted order (0 < 1, but 0 is at i and 1 is at j, with i < j).
        # So yes, state[j] must be a subset of state[i] for all i < j?
        # Wait.
        # Sorted ascending: 0, 0, 1, 1.
        # v = [0, 0, 1, 1].
        # state[0] bit k = 0.
        # state[1] bit k = 0.
        # state[2] bit k = 1.
        # state[3] bit k = 1.
        # Condition: state[j] & ~state[i] == 0 for i < j.
        # i=0, j=2: state[2] (1) & ~state[0] (1) = 1. Non-zero.
        # So my condition is wrong.
        
        # Let's re-evaluate.
        # v sorted ascending: v[0] <= v[1] <= ...
        # If v[j] == 1, then v[i] must be 1 for all i > j? No, v[i] >= v[j].
        # If v[j] == 1, then v[i] must be 1 for all i > j?
        # Yes, because 1 is the max. If v[j]=1, all subsequent must be 1.
        # If v[i] == 0, then v[j] must be 0 for all j < i?
        # Yes, because 0 is the min. If v[i]=0, all previous must be 0.
        
        # So:
        # If state[j] has bit k (v_k[j]=1), then state[m] must have bit k for all m > j.
        # This means state[j] must be a SUPERSET of state[m] for all m > j?
        # No. state[j] has bit k. state[m] must have bit k.
        # So state[m] & ~state[j] == 0? No.
        # If state[j] has bit k, state[m] must have bit k.
        # So (state[m] & ~state[j]) is not the check.
        # The check is: if state[j] has bit k, state[m] MUST have bit k.
        # This is equivalent to: state[j] is a subset of state[m] for all m > j?
        # No. If state[j] has bit k, state[m] has bit k.
        # So state[j] & ~state[m] == 0 for all m > j.
        
        # Let's test with v=[0,0,1,1].
        # state[2] has bit k. state[3] has bit k.
        # state[2] & ~state[3] = 1 & ~1 = 0. Correct.
        # state[0] has no bit k. state[2] has bit k.
        # state[0] & ~state[2] = 0 & ~1 = 0. Correct.
        
        # So the condition is: for all i < j, state[i] & ~state[j] == 0.
        # This means state[i] is a subset of state[j].
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    current_network = best_network[:]
    current_cost = len(current_network)
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    
    def try_remove_all(current_net):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        while removed:
            if time.time() > deadline:
                break
            removed = False
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    def grow_and_shrink(current_net, num_adds):
        """Add random comparators, then aggressively remove redundancies."""
        candidate = current_net[:]
        for _ in range(num_adds):
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return None
        
        optimized = try_remove_all(candidate)
        return optimized

    # Main Search Loop
    while time.time() < deadline:
        # 1. Aggressively reduce current network
        optimized = try_remove_all(current_network)
        if len(optimized) < current_cost:
            current_network = optimized
            current_cost = len(optimized)
            if current_cost < best_cost:
                best_network = current_network[:]
                best_cost = current_cost
        
        if time.time() > deadline:
            break

        # 2. Mutation Phase
        r = random.random()
        if r < 0.5:
            num_adds = 1
        elif r < 0.8:
            num_adds = 2
        elif r < 0.95:
            num_adds = 3
        else:
            num_adds = 4
            
        new_net = grow_and_shrink(current_network, num_adds)
        
        if new_net is not None:
            new_cost = len(new_net)
            delta = new_cost - current_cost
            
            if delta <= 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / T)
                accept = random.random() < accept_prob
            
            if accept:
                current_network = new_net
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
        
        T = max(T * cooling_rate, T_min)

    return best_network
# EVOLVE-BLOCK-END
