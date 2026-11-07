import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    n = len(coordinates)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base inverse distance
    d_safe = np.where(dm > 0, dm, 1e-9)
    base = 1.0 / (d_safe + 1e-9)
    np.fill_diagonal(base, 1e-9)
    
    # Build Delaunay triangulation and identify edges
    try:
        tri = Delaunay(coords)
    except Exception:
        return base
    
    delaunay_edges = set()
    for simplex in tri.simplices:
        i, j, k = simplex
        delaunay_edges.add((i, j))
        delaunay_edges.add((j, i))
        delaunay_edges.add((j, k))
        delaunay_edges.add((k, j))
        delaunay_edges.add((k, i))
        delaunay_edges.add((i, k))
    
    # Build Voronoi tessellation for capacity-aware spatial partitioning
    voronoi_cells = None
    try:
        vor = Voronoi(coords)
        voronoi_cells = vor.regions
        vor_points = vor.points
        vor_vertices = vor.vertices
    except Exception:
        voronoi_cells = None
    
    # Compute per-node local capacity density and cell radius
    # Node 0 is depot with demand 0
    node_density = np.zeros(n)
    cell_radii = np.zeros(n)
    
    if voronoi_cells is not None:
        for i in range(n):
            region = voronoi_cells[i]
            if len(region) == 0:
                continue
            verts = vor_vertices[region]
            # Filter out infinite vertices (marked as -1)
            finite_verts = verts[np.all(verts >= 0, axis=1)]
            if len(finite_verts) < 2:
                continue
            
            # Compute minimum enclosing circle radius approximation using
            # the maximum distance from centroid to any vertex
            centroid = finite_verts.mean(axis=0)
            dists = np.linalg.norm(finite_verts - centroid, axis=1)
            max_dist = dists.max()
            
            # Local density: sum of demands of nodes whose Voronoi cells are adjacent
            # Approximate by considering Delaunay neighbors
            neighbor_demands = 0.0
            for (ni, nj) in delaunay_edges:
                if ni == i:
                    neighbor_demands += dem[nj]
                    break
            
            # Capacity density ratio: how much demand is packed in this local area
            # Higher density means tighter local constraints
            area_proxy = max_dist ** 2 if max_dist > 0 else 1e-9
            local_demand = dem[i] + neighbor_demands * 0.3
            node_density[i] = local_demand / area_proxy
            
            cell_radii[i] = max_dist
    
    # Fallback: if Voronoi failed, use k-NN based local density
    if voronoi_cells is None or np.all(cell_radii == 0):
        try:
            tree = cKDTree(coords)
            # Get 5 nearest neighbors for each node
            _, nn_indices = tree.query(coords, k=6)  # includes self
            for i in range(n):
                neighbors = nn_indices[i, 1:]  # exclude self
                neighbor_demands = dem[neighbors].sum()
                neighbor_dists = dm[i, neighbors]
                avg_dist = neighbor_dists.mean() if len(neighbor_dists) > 0 else 1.0
                local_demand = dem[i] + neighbor_demands * 0.3
                node_density[i] = local_demand / (avg_dist ** 2 + 1e-9)
                cell_radii[i] = avg_dist
        except Exception:
            pass
    
    # Apply heuristic modifications
    heuristic = base.copy()
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # 1. Penalize non-Delaunay edges
            if (i, j) not in delaunay_edges:
                heuristic[i, j] *= 0.01
            
            # 2. Capacity-aware Voronoi penalty
            # Penalize edges that span very different local density regions
            # or that are long relative to the shared cell boundary
            if i < n and j < n and cell_radii[i] > 0 and cell_radii[j] > 0:
                # Edge length relative to the minimum enclosing circle of the shared region
                min_radius = min(cell_radii[i], cell_radii[j])
                edge_len = dm[i, j]
                
                # Ratio: if edge is much longer than local cell size, penalize
                if min_radius > 1e-6:
                    ratio = edge_len / min_radius
                    # Smooth penalty: log-sum-exp style soft penalty
                    # For ratio > 1, apply exponential decay
                    penalty = np.exp(-0.5 * max(0, ratio - 1.0))
                    heuristic[i, j] *= penalty
                
                # Density mismatch penalty: connecting very dense to very sparse areas
                # is less efficient for capacity-constrained routing
                if node_density[i] > 0 and node_density[j] > 0:
                    density_ratio = max(node_density[i], node_density[j]) / (min(node_density[i], node_density[j]) + 1e-9)
                    # Soft penalty for large density mismatches
                    density_penalty = 1.0 / (1.0 + 0.1 * (density_ratio - 1.0) ** 2)
                    heuristic[i, j] *= density_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic
