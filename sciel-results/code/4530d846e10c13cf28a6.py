# Accept or reject
if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
    # Accept the move
    chain[i] = new_pos
    where[new_pos] = i
    current_energy += delta_e
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_chain = list(chain)
    
    moved = True
    break
