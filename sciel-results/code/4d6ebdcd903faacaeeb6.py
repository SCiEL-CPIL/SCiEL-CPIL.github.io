import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses a local search strategy with vectorized validity checks.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 20

    # Pre-allocate arrays
    adj = np.zeros((n, n), dtype=np.int8)
    common = np.zeros((n, n), dtype=np.int8)

    def update_common_add(u, v):
        """Update common neighbor matrix after adding edge (u, v)."""
        # Get neighbors of u and v BEFORE adding the edge
        # Note: u is not in N(v) and v is not in N(u) currently
        nu = np.nonzero(adj[u])[0]
        nv = np.nonzero(adj[v])[0]

        # Add the edge
        adj[u, v] = 1
        adj[v, u] = 1

        # Update common counts
        # For w in N(v), common[u, w] increases by 1
        if len(nv) > 0:
            common[u, nv] += 1
            common[nv, u] += 1
        # For w in N(u), common[v, w] increases by 1
        if len(nu) > 0:
            common[v, nu] += 1
            common[nu, v] += 1

    def update_common_remove(u, v):
        """Update common neighbor matrix after removing edge (u, v)."""
        # Get neighbors of u and v BEFORE removing the edge
        # Note: u IS in N(v) and v IS in N(u) currently
        nu = np.nonzero(adj[u])[0]
        nv = np.nonzero(adj[v])[0]

        # Remove the edge
        adj[u, v] = 0
        adj[v, u] = 0

        # Update common counts
        # Only update for w != u and w != v
        mask_nv = nv[nv != u]
        mask_nu = nu[nu != v]

        if len(mask_nv) > 0:
            common[u, mask_nv] -= 1
            common[mask_nv, u] -= 1
        if len(mask_nu) > 0:
            common[v, mask_nu] -= 1
            common[mask_nu, v] -= 1

    def can_add_edge(u, v):
        """Check if edge (u, v) can be added."""
        if adj[u, v]:
            return False
        if common[u, v] > 2:
            return False

        nv = np.nonzero(adj[v])[0]
        if len(nv) > 0:
            # Check max common[u, w] for w in N(v)
            if np.max(common[u, nv]) > 1:
                return False

        nu = np.nonzero(adj[u])[0]
        if len(nu) > 0:
            # Check max common[v, w] for w in N(u)
            if np.max(common[v, nu]) > 1:
                return False

        return True

    def greedy_add():
        """Greedily add edges until no more can be added."""
        added = True
        while added:
            added = False
            # Randomize order to avoid bias
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add_edge(u, v):
                    update_common_add(u, v)
                    added = True
                    # Don't break, continue to add more if possible
        return int(np.sum(adj) // 2)

    def is_valid():
        """Check if the graph is K2,3-free."""
        mask = ~np.eye(n, dtype=bool)
        return np.max(common[mask]) <= 2

    def repair():
        """Remove edges to make the graph valid."""
        changed = True
        while changed:
            changed = False
            # Find violations
            # common[u, v] > 2
            # We need to remove an edge (u, w) or (v, w) where w is a common neighbor.
            for u in range(n):
                for v in range(u + 1, n):
                    if common[u, v] > 2:
                        # Find common neighbors
                        cn = np.nonzero(adj[u] & adj[v])[0]
                        if len(cn) > 0:
                            # Remove a random edge from one of the common neighbors
                            w = random.choice(cn.tolist())
                            # Remove (u, w)
                            update_common_remove(u, w)
                            changed = True
                            break
                if changed:
                    break
        return is_valid()

    def get_edge_list():
        """Get list of edges from adj matrix."""
        return [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]

    def get_non_edge_list():
        """Get list of non-edges from adj matrix."""
        return [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]

    best_count = 0
    best_adj = None
    best_common = None

    # Use multiple restarts
    for seed in range(500):
        if time.time() > deadline:
            break

        random.seed(seed)
        np.random.seed(seed)

        # Initialize
        adj.fill(0)
        common.fill(0)

        # Start with a random sparse graph
        # Probability of edge ~ 0.4
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.4:
                    adj[u, v] = 1
                    adj[v, u] = 1

        # Recalculate common
        common = (adj @ adj).astype(np.int8)

        # Repair
        repair()

        # Greedy add
        greedy_add()

        current_count = int(np.sum(adj) // 2)

        if current_count > best_count:
            best_count = current_count
            best_adj = adj.copy()
            best_common = common.copy()

        # Local Search with kick mechanism
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        no_improve_steps = 0
        kick_threshold = 200  # If no improvement for this many steps, kick

        while time.time() < deadline and T > min_T:
            # Pick a random move
            move_type = random.random()

            if move_type < 0.6:
                # Try to add a random edge
                u, v = random.sample(range(n), 2)
                if not adj[u, v] and can_add_edge(u, v):
                    update_common_add(u, v)
                    current_count += 1
                    no_improve_steps = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj.copy()
                        best_common = common.copy()
                else:
                    no_improve_steps += 1
            else:
                # Try to swap: Remove a random edge, then try to add a random edge
                edges = get_edge_list()
                if edges:
                    ru, rv = random.choice(edges)
                    # Backup
                    adj_backup = adj.copy()
                    common_backup = common.copy()

                    update_common_remove(ru, rv)
                    current_count -= 1

                    # Try to add a random non-edge
                    added = False
                    for _ in range(15):
                        nu, nv = random.sample(range(n), 2)
                        if not adj[nu, nv] and can_add_edge(nu, nv):
                            update_common_add(nu, nv)
                            current_count += 1
                            added = True
                            break

                    if not added:
                        # Revert if we didn't add anything back
                        adj = adj_backup
                        common = common_backup
                        current_count += 1
                    else:
                        # Check if we improved best
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj.copy()
                            best_common = common.copy()
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1

            # Kick mechanism: if stuck, remove a random edge to explore
            if no_improve_steps > kick_threshold:
                edges = get_edge_list()
                if edges:
                    ru, rv = random.choice(edges)
                    update_common_remove(ru, rv)
                    current_count -= 1
                    no_improve_steps = 0

            T *= cooling

    if best_adj is None:
        return []

    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])

    return best_edges
