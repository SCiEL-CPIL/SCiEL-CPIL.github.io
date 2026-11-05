import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute a composite score based on degree (prefer low degree) and neighbor degree.
    2. Use an iterative "simulated greedy" refinement:
       - Simulate the greedy selection process.
       - At each step, evaluate the marginal gain of selecting each remaining vertex.
       - Adjust scores based on how much "coverage" each vertex provides.
    3. Use eigenvector centrality as a tie-breaker or secondary feature.
    4. Deadline guard to ensure we finish within 120s.
    """
    start_time = time.time()
    DEADLINE = 100.0  # Safety margin before 120s limit
    
    try:
        adj = np.asarray(adjacency, dtype=float)
        n = adj.shape[0]
        
        if n < 2:
            return np.zeros(n)
        
        # Compute degrees
        deg = adj.sum(axis=1)
        
        # Compute neighbor degree (sum of degrees of neighbors)
        # This is a good proxy for "how much damage" selecting this vertex does
        # In MIS, we want to select vertices that don't block too many others
        # So we prefer vertices with LOW degree AND LOW neighbor degree
        
        # neighbor_deg[i] = sum of deg[j] for j in neighbors of i
        neighbor_deg = adj @ deg
        
        # Composite score: we want to select vertices that are "cheap" to include
        # Low degree means fewer neighbors are blocked
        # Low neighbor degree means the neighbors we block are less important
        # So score = -(degree + alpha * neighbor_degree)
        # Higher score = more desirable
        
        alpha = 0.5
        base_score = -(deg + alpha * neighbor_deg)
        
        # Add eigenvector centrality as a tie-breaker
        # Eigenvector centrality: vertices connected to high-centrality vertices are more central
        # For MIS, we might want to AVOID high-centrality vertices in the core
        # But we can use it as a subtle signal
        
        # Compute eigenvector centrality using power iteration
        # x_{k+1} = A @ x_k / ||A @ x_k||
        try:
            x = np.ones(n)
            for _ in range(20):
                x_new = adj @ x
                norm = np.linalg.norm(x_new)
                if norm < 1e-10:
                    break
                x_new = x_new / norm
                if np.linalg.norm(x_new - x) < 1e-8:
                    x = x_new
                    break
                x = x_new
            
            # Subtract mean to center
            x_centered = x - np.mean(x)
            
            # Use as a small additive term
            # Vertices with high eigenvector centrality are in the "core"
            # We slightly penalize them (lower score) because they're more constrained
            score = base_score - 0.1 * x_centered
            
        except Exception:
            score = base_score.copy()
        
        # Iterative refinement: simulate greedy and adjust
        # This is the key structural improvement: dynamic, state-dependent scoring
        
        deadline_check = time.time() - start_time
        if deadline_check > DEADLINE * 0.5:
            # Not enough time for refinement, return current score
            return score
        
        # Simulate greedy selection
        remaining = set(range(n))
        selected = []
        
        # We'll do a few passes of "what-if" analysis
        # For each vertex, estimate its marginal contribution if selected now
        # Marginal contribution = 1 (itself) - penalty for blocking neighbors
        
        # Simple approach: iterate and adjust scores based on local structure
        for iteration in range(3):
            if time.time() - start_time > DEADLINE:
                break
            
            # Recompute scores based on remaining vertices
            # Vertices that are neighbors of many high-score vertices should be penalized
            # Vertices that are isolated or have few high-score neighbors should be boosted
            
            # Compute "conflict" score: how many high-scoring neighbors each vertex has
            # High conflict = should be selected later (lower priority)
            
            # Sort by current score
            order = np.argsort(-score)
            
            # For each vertex, count how many of its neighbors have higher score
            conflict = np.zeros(n)
            for i in range(n):
                v = order[i]
                if v not in remaining:
                    continue
                # Count neighbors with higher score
                neighbors = np.where(adj[v] > 0)[0]
                for nb in neighbors:
                    if nb in remaining and score[nb] > score[v]:
                        conflict[v] += 1
            
            # Penalize high-conflict vertices
            # But also consider degree: a vertex with high conflict but low degree is still good
            adjusted_penalty = conflict * (1.0 / (deg + 1))
            
            # Update score: reduce score for high-conflict vertices
            score = score - 0.3 * adjusted_penalty
            
            # Renormalize to keep scores in a reasonable range
            score = score - np.mean(score)
        
        # Final safety check
        if time.time() - start_time > DEADLINE:
            # Return the base score if we're running out of time
            return base_score
        
        return score
    
    except Exception:
        # Fallback to simple degree-based score
        try:
            deg = np.sum(adjacency, axis=1)
            return -deg
        except:
            return np.zeros(adjacency.shape[0])
