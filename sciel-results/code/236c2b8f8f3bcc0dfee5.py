import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
        d = np.array(distance_matrix, dtype=float)
        coords = np.array(coordinates, dtype=float)
        dem = np.array(demands, dtype=float)
        n = d.shape[0]

        # Sharpened inverse-distance core (kept from current design)
        d_safe = d + 1e-9
        dist_heur = 1.0 / (d_safe ** 3)

        # Demand-aware component: prefer customers whose demand is a moderate
        # fraction of remaining capacity.  For a fresh load (full capacity) a
        # demand near capacity/2 packs well; very small demands are fine too,
        # very large (close to capacity) leave little room for further stops.
        # We model "remaining capacity" as the full capacity for this static
        # heuristic (the colony re-computes nothing; this is a fixed prior).
        # Use a soft preference that peaks at mid-range demand.
        cap = float(capacity)
        # demand ratio relative to capacity
        ratio = dem / cap  # shape (n,), depot demand 0 -> ratio 0
        # A smooth bump that is high for ratio in [0.2, 0.6] and tapers off.
        # Use a Gaussian-like log-sum-exp-free smooth form.
        center = 0.35
        width = 0.25
        demand_factor = np.exp(-((ratio - center) ** 2) / (2.0 * width * width))
        # Depot: keep neutral (1.0) so returning-to-depot isn't over-suppressed
        demand_factor[0] = 1.0
        # Broadcast: heuristic[i,j] *= demand_factor[j] (the destination's demand)
        demand_matrix = demand_factor[np.newaxis, :]

        # Combine multiplicatively: distance sharpness * destination demand fit
        heuristic = dist_heur * demand_matrix

        # Set diagonal to 0 to discourage self-loops if the ACO doesn't mask them
        np.fill_diagonal(heuristic, 0.0)

        # Ensure any non-positive values (if d was somehow negative, unlikely) are 1e-9
        heuristic[heuristic <= 0] = 1e-9

        return heuristic
