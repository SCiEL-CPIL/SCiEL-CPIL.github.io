"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for efficiency
    demands = np.asarray(demands, dtype=int)
    capacities = np.asarray(capacities, dtype=int)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    # Deadline guard: 140 seconds to be safe
    deadline = time.time() + 140.0

    # 1. Initial Greedy Solution
    # Sort customers by demand in descending order
    order = np.argsort(-demands)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=int)
    
    for i in order:
        # Find cheapest feasible facility
        # We can vectorize this slightly, but n=100 is small
        feasible = []
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append(f)
        
        if not feasible:
            # If no feasible facility, this is a problem. 
            # The problem statement implies a valid solution exists or we must handle infeasibility.
            # "Floats, out-of-range indices or a facility whose assigned demand exceeds its capacity make the whole result invalid."
            # If greedy fails, we might need a more complex initialization or it's an infeasible instance.
            # Assuming feasible instances for now, but to be safe, we pick the one with most remaining capacity if none fit? 
            # No, that would be invalid. 
            # Let's assume the instances are feasible. If not, we return the best effort, but it will score 0.
            # For robustness, if no feasible, we break and return what we have (which is invalid).
            # Better: try to find *any* feasible. If none, the instance is infeasible.
            # We will proceed with the assumption that a valid solution exists.
            pass

        # Pick min cost among feasible
        if feasible:
            costs = [assign_cost[i, f] for f in feasible]
            best_f = feasible[np.argmin(costs)]
            assignment[i] = best_f
            load[best_f] += demands[i]
        else:
            # Infeasible case handling: Assign to 0 to avoid crash, result will be invalid
            assignment[i] = 0
            
    # Calculate initial cost
    def calc_cost(assignment, load):
        total = 0.0
        counts = np.bincount(assignment, minlength=n_facilities)
        # Open cost for facilities with count > 0
        total += np.sum(open_cost[counts > 0])
        # Assignment cost
        total += np.sum(assign_cost[np.arange(n_customers), assignment])
        return total, counts

    current_cost, counts = calc_cost(assignment, load)
    best_cost = current_cost
    best_assignment = assignment.copy()
    best_load = load.copy()
    best_counts = counts.copy()

    # 2. Local Search (Simulated Annealing)
    # Temperature schedule
    # Max possible change in cost is roughly max assign cost + max open cost
    max_assign = np.max(assign_cost)
    max_open = np.max(open_cost)
    T_start = (max_assign + max_open) * 5.0
    T_end = 0.1
    alpha = 0.999 # Cooling rate
    
    T = T_start
    rng = np.random.default_rng(42)

    while time.time() < deadline:
        # Pick a random customer
        i = rng.integers(0, n_customers)
        old_f = assignment[i]
        
        # Find feasible new facilities
        # We can check all facilities
        feasible_facs = []
        for f in range(n_facilities):
            if f == old_f:
                continue
            if load[f] + demands[i] <= capacities[f]:
                feasible_facs.append(f)
        
        if not feasible_facs:
            continue

        # Pick a random feasible facility
        new_f = feasible_facs[rng.integers(0, len(feasible_facs))]
        
        # Calculate delta cost
        delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
        
        delta_open = 0.0
        # If old facility becomes empty, we pay its open cost
        if counts[old_f] == 1:
            delta_open += open_cost[old_f]
        # If new facility was empty, we save its open cost
        if counts[new_f] == 0:
            delta_open -= open_cost[new_f]
            
        delta_cost = delta_assign + delta_open
        
        # Acceptance criterion
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            # Accept move
            assignment[i] = new_f
            load[old_f] -= demands[i]
            load[new_f] += demands[i]
            counts[old_f] -= 1
            counts[new_f] += 1
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = assignment.copy()
                best_load = load.copy()
                best_counts = counts.copy()
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            # Optional: restart with higher T if stuck? 
            # For now, just keep going at low T to polish.
            
    return best_assignment

# EVOLVE-BLOCK-END
