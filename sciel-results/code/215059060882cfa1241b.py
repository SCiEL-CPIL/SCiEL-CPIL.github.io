def fold(sequence, dimension):
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    initial_chains = []
    growth_budget = 20
    biases = [1.0, 1.5, 2.0, 0.5, 0.8]
    for trial in range(growth_budget):
        if _remaining_time() < 10:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    time_per_tabu = max(1.0, _remaining_time() / max(1, len(initial_chains)) * 0.8)
    
    for idx, chain in enumerate(initial_chains[:5]):
        if _remaining_time() < 5:
            break
        
        t = time_per_tabu
        if idx == 0:
            t *= 2.0
            
        annealed_chain, annealed_energy = _tabu_search(sequence, chain, max_time=t, seed=12345 + idx)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    if _remaining_time() > 3 and best is not None:
        polish_time = min(2.0, _remaining_time() - 1)
        best, best_energy = _polish(sequence, best, max_time=polish_time, seed=99999)

    return [list(p) for p in best]
