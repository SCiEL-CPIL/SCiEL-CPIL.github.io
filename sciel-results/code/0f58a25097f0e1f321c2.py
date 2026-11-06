<<<<<<< SEARCH
def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best = list(seed_network)
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    # 1. Initial Greedy reduction to get a compact baseline
    best = _greedy_reduce(best, deadline)
    
    # 2. Simulated Annealing loops
    while time.time() < deadline:
        # Decide starting point for SA
        if random.random() < 0.6:
            # Local search: start from best
            start_net = list(best)
        else:
            # Global search: start from seed, perturb, reduce
            start_net = list(seed_network)
            # Add some random noise to break symmetry
            num_noise = random.randint(5, 15)
            for _ in range(num_noise):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            
            # Reduce the perturbed seed to make it a reasonable starting point
            reduce_deadline = min(deadline, time.time() + 5)
            start_net = _greedy_reduce(start_net, reduce_deadline)
            
            if time.time() > deadline:
                break
        
        # Run SA
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            if time.time() < deadline:
                quick_deadline = min(deadline, time.time() + 2)
                best = _greedy_reduce(best, quick_deadline)
                
    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin before 60s limit
    
    # Build candidate starting networks
    oet = _odd_even_transposition()
    batcher = _batcher_18()
    
    # Start from the denser Batcher network (more redundancy to remove)
    best = _greedy_reduce(batcher, min(deadline, start_time + 15))
    
    # Also try reducing the OET seed
    oet_reduced = _greedy_reduce(oet, min(deadline, start_time + 5))
    if len(oet_reduced) < len(best):
        best = oet_reduced
    
    # Simulated Annealing loops
    while time.time() < deadline:
        r = random.random()
        if r < 0.5:
            # Local search: start from best
            start_net = list(best)
        elif r < 0.8:
            # Global search from Batcher with perturbation
            start_net = list(batcher)
            num_noise = random.randint(3, 10)
            for _ in range(num_noise):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            reduce_deadline = min(deadline, time.time() + 4)
            start_net = _greedy_reduce(start_net, reduce_deadline)
        else:
            # Global search from OET with perturbation
            start_net = list(oet)
            num_noise = random.randint(5, 15)
            for _ in range(num_noise):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            reduce_deadline = min(deadline, time.time() + 4)
            start_net = _greedy_reduce(start_net, reduce_deadline)
        
        if time.time() > deadline:
            break
        
        # Run SA
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            if time.time() < deadline:
                quick_deadline = min(deadline, time.time() + 2)
                best = _greedy_reduce(best, quick_deadline)
    
    # Final high-precision polish: bidirectional greedy removal
    remaining = deadline - time.time()
    if remaining > 2:
        best = _greedy_reduce_bidirectional(best, deadline - 1)
    
    return best
>>>>>>> REPLACE
