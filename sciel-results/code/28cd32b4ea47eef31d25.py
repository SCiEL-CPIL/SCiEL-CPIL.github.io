"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Combined Heuristic:
    1. Primary: Prefer vertices with lower degree (fewer neighbors to block).
    2. Secondary: Prefer vertices whose neighbors have higher average degrees.
       Rationale: Picking a vertex 'v' blocks its neighbors. If the blocked
       neighbors are high-degree "hubs", it is beneficial because those hubs
       would have blocked many other vertices if they had been picked.
       By blocking them early, we 'save' their neighbors from being blocked.
    3. Tertiary (Spectral): Use the Fiedler vector (second smallest eigenvector
       of the Laplacian) to identify global structural partitioning. Vertices
       on one side of the sparsest cut may be prioritized to form a coherent
       independent set component, reducing fragmentation.
    4. Quaternary (Triangle): Prefer vertices in dense clusters (triangles).
       Blocking a triangle efficiently removes 3 vertices and their connections.
       However, since we want to *pick* vertices, we actually want to avoid
       picking vertices that are in dense clusters if it blocks many others,
       OR we want to pick them to quickly clear the cluster. 
       Wait, the inspiration 2 added triangle count positively. 
       Let's think: if v is in a triangle (v,u,w), picking v blocks u and w.
       If u and w are also high degree, this is good. 
       The term 0.5 * tri_count in inspiration 2 suggests it helped slightly (1.015 vs 1.0075 base?).
       Actually, inspiration 1 (1.0263) is just -deg + mean_deg.
       Inspiration 2 (1.0150) added triangles and spectral.
       Inspiration 1 is the current best.
       
       Let's look at the "Meta-heuristic search" note. It says local search around good static features can escape plateaus.
       But we are limited to a single static function call.
       
       Let's refine the base heuristic.
       The current best is: -degrees + mean_neighbor_deg.
       
       Let's try to incorporate a "coreness" or "closeness" aspect or refine the neighbor term.
       Another idea: "Weighted" degree.
       
       Let's try a different spectral approach: 
       Instead of just the Fiedler vector, use the eigenvector corresponding to the *largest* eigenvalue (principal eigenvector)? 
       No, Fiedler is for cuts.
       
       What if we use a "soft" greedy simulation?
       We can't run the full greedy simulation because the decoder does that.
       But we can approximate the "regret" of picking a vertex.
       
       Let's stick to the strongest inspiration: Inspiration 1 is the current best.
       Inspiration 2 was worse.
       Inspiration 3 was worse.
       
       The directive says: "Use a spectral gap heuristic... prioritize vertices in the smaller partition".
       The current program uses Fiedler vector with alpha=1.0.
       
       Let's try to tune the spectral weight or use a different spectral feature.
       Or, combine the "mean neighbor degree" with "triangle count" but with a negative weight?
       If a vertex is in many triangles, its neighbors are likely connected to each other.
       Picking it blocks neighbors that are already "clustered".
       Maybe blocking a hub in a sparse part is better than blocking a hub in a dense part?
       
       Let's try: Score = -degree + mean_neighbor_deg - beta * triangle_count.
       If we subtract triangle count, we deprioritize vertices in dense clusters.
       Why? Because in a dense cluster, the independent set size is small (clique number).
       Picking a vertex in a dense cluster yields low "value" per blocked neighbor.
       Picking a vertex in a sparse area yields higher "value".
       
       Let's test this hypothesis mentally.
       ER(200, p) graphs.
       If p is high, graph is dense.
       If p is low, graph is sparse.
       
       Let's try:
       Score = -degree + mean_neighbor_deg - 0.1 * triangle_count
       
       Also, let's refine the spectral part.
       Instead of just adding the Fiedler vector, maybe we should use the sign of the Fiedler vector to create a binary split and prioritize one side?
       No, that's too coarse.
       
       Let's try to combine the "Meta-heuristic" insight.
       The insight says: "local search around good static features".
       We can't do local search on the instance in the function call if we want to stay fast, but we *can* do a quick "greedy simulation" to estimate the marginal gain of each vertex.
       
       Algorithm:
       1. Compute base scores S_i = -deg(i) + mean_deg(i).
       2. Sort vertices by S_i.
       3. Simulate the greedy selection using this order.
       4. For each vertex v, calculate how many *other* vertices would have been selected if v was NOT selected (but its neighbors were blocked by other means?). This is complex.
       
       Simpler Approximation:
       1. Compute base scores.
       2. Run a quick greedy pass to identify which vertices are "critical" (picked early) and which are "wasted" (blocked by earlier picks).
       3. Boost the score of vertices that are "efficient" blockers (high degree neighbors, low own degree) AND are not in dense clusters.
       
       Actually, let's look at the "Spectral Gap" directive again.
       "Prioritize vertices in the smaller partition to minimize the number of edges crossing the independent set boundary."
       
       Let's try a weighted combination:
       Base = -degree + mean_neighbor_deg
       Spectral = Fiedler vector (normalized)
       Triangle = Triangle count
       
       New Score = Base + alpha * Spectral - beta * Triangle
       
       Current best (Inspiration 1) is just Base. Score 1.0263.
       Inspiration 2 (Base + 0.5*Tri + 0.5*Spec) was 1.0150.
       So adding Tri and Spec *hurt* performance in Inspiration 2.
       
       Why? Maybe the weights were wrong, or the features were redundant/noisy.
       
       Let's try *subtracting* the triangle count.
       Score = Base - 0.1 * Triangle
       
       And let's try a *smaller* spectral weight, or remove it if it hurts.
       Since Inspiration 2 (with spectral) was worse than Inspiration 1 (without), spectral might be noise.
       
       Let's try: Score = -degree + mean_neighbor_deg - 0.5 * (triangle_count / max_triangle)
       
       Or perhaps the "mean neighbor degree" term needs refinement.
       What if we use the *max* neighbor degree?
       Or the *second max*?
       
       Let's try a "weighted" neighbor degree sum where neighbors are weighted by their inverse degree?
       No, that's getting complicated.
       
       Let's try a simple modification to the base:
       Score = -degree + mean_neighbor_deg - 0.1 * triangle_count
       
       And let's keep the spectral part but with a very small weight, or remove it to match the best inspiration structure (Inspiration 1) but add the triangle penalty.
       
       Let's calculate triangle count efficiently.
       A^2 = adj @ adj
       Tri_count = 0.5 * sum(adj * A^2, axis=1)
       
       Let's try:
       priorities = -degrees + mean_neighbor_deg - 0.2 * tri_count
       
       Why 0.2?
       Tri_count can be up to ~100s for dense graphs.
       Degree is ~10-50.
       Mean neighbor deg is ~10-50.
       So 0.2 * 100 = 20, which is comparable to degree.
       
       Let's also consider the "Spectral" directive.
       If I ignore the directive, I might miss a jump.
       The directive says: "prioritize vertices in the smaller partition".
       
       Let's try:
       1. Compute Fiedler vector.
       2. Split vertices into two sets based on sign of Fiedler vector.
       3. If the smaller set has a lower average degree, prioritize it?
       4. Or simply add the Fiedler vector with a sign that favors the "sparser" side?
       
       Actually, the Fiedler vector values are proportional to the "distance" from the cut.
       Vertices with high absolute values are "core" of their partition.
       Vertices with low absolute values are "boundary".
       We want to pick "core" vertices to form a stable independent set?
       Or do we want to pick "boundary" vertices to block the cut?
       
       If we pick a core vertex, we block its neighbors (which are likely in the same partition).
       If we pick a boundary vertex, we block neighbors in both partitions.
       
       For an independent set, we want to maximize the number of picked vertices.
       In a bipartite graph, the max independent set is the larger partition.
       The Fiedler vector for a bipartite graph has +/- 1 values.
       If we prioritize vertices in the larger partition, we get a larger IS.
       
       So, we should check the size of the two partitions defined by the Fiedler vector sign.
       If partition A is larger than partition B, we should boost vertices in A.
       
       How to do this in the score?
       If |fiedler[i]| > 0, we know the side.
       Let's count how many vertices are positive vs negative.
       If positive_count > negative_count, we want to boost positive vertices.
       So we add: sign(fiedler[i]) * (positive_count - negative_count) / n * weight?
       
       This is a "partition bias" term.
       
       Let's implement this:
       1. Compute Base = -deg + mean_deg.
       2. Compute Tri = triangle count.
       3. Compute Fiedler vector.
       4. Determine the majority side of the Fiedler split.
       5. Add a bias term that favors the majority side.
       6. Subtract a small penalty for triangle count.
       
       Score = Base - 0.1 * Tri + 0.5 * sign(fiedler[i]) * bias_factor
       
       Where bias_factor = (abs(pos_count - neg_count)) / n
       
       This encourages picking from the larger "side" of the graph's main cut, which often correlates with the larger independent set in bipartite-like structures.
       
       Let's code this.
       
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate mean degree of neighbors
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1.0)
    
    # Base priority
    base_priorities = -degrees + mean_neighbor_deg
    
    # Calculate Triangle Count
    # A^2 counts paths of length 2.
    # (A * A^2)_{vu} is non-zero only if A_{vu}=1, in which case it is the number of common neighbors.
    # Summing over u gives twice the number of triangles containing v.
    A2 = adj @ adj
    tri_sum = (adj * A2).sum(axis=1).astype(np.float64)
    tri_count = 0.5 * tri_sum
    
    # Penalty for being in dense clusters (triangles)
    # Scale: tri_count can be ~100. Base is ~-20 to 20.
    # Use a small weight to deprioritize dense areas slightly.
    tri_penalty = 0.1 * tri_count
    
    current_priorities = base_priorities - tri_penalty
    
    # Deadline guard
    if time.time() - start_time > 100:
        return current_priorities.astype(float)
        
    # Spectral Heuristic: Partition Bias
    try:
        D = np.diag(degrees)
        L = D - adj.astype(np.float64)
        
        if time.time() - start_time > 110:
            return current_priorities.astype(float)
            
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Find the Fiedler vector (first non-zero eigenvalue's eigenvector)
        non_zero_idx = np.where(eigvals > 1e-8)[0]
        if len(non_zero_idx) > 0:
            fiedler_idx = non_zero_idx[0]
        else:
            fiedler_idx = 1
            
        fiedler_vec = eigvecs[:, fiedler_idx]
        
        # Determine the partition sizes
        # Vertices with fiedler_vec > 0 are in partition A, < 0 in partition B
        pos_mask = fiedler_vec > 0
        neg_mask = fiedler_vec < 0
        
        pos_count = np.sum(pos_mask)
        neg_count = np.sum(neg_mask)
        
        # Bias towards the larger partition
        # If pos_count > neg_count, we want to boost positive vertices.
        # Bias value = (pos_count - neg_count) / n
        # This is in range [-1, 1]
        bias = (pos_count - neg_count) / n
        
        # Create a bias vector: +bias for pos, -bias for neg
        # This effectively adds 'bias' to the larger side and subtracts from the smaller side.
        # Wait, if bias is positive (pos > neg), we add bias to pos vertices and subtract bias from neg vertices?
        # Yes, sign(fiedler_vec) * bias.
        # If fiedler_vec > 0, term is +bias.
        # If fiedler_vec < 0, term is -bias.
        
        spectral_bias = np.sign(fiedler_vec) * bias
        
        # Weight for the spectral bias
        # The base priorities are in range ~[-50, 50].
        # Bias is in [-1, 1].
        # Let's use a weight of 5.0 to make it significant but not dominant.
        alpha_spectral = 5.0
        
        current_priorities += alpha_spectral * spectral_bias
        
    except Exception:
        # If spectral computation fails, fall back to current priorities
        pass
        
    # Final deadline guard
    if time.time() - start_time > 115:
        return current_priorities.astype(float)
        
    return current_priorities.astype(float)
# EVOLVE-BLOCK-END
