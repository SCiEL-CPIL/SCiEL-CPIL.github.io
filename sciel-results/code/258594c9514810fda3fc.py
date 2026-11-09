"""Rewritten program for Maximum Cut using Spectral Initialization and Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Vectorized calculation: (labels[:, None] != labels[None, :]) & adj
    # Sum and divide by 2 because each edge is counted twice (i,j) and (j,i)
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2

def _spectral_init(adj):
    """
    Initialize labels using the Fiedler vector of the graph Laplacian.
    The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
    """
    n = len(adj)
    # Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Compute eigenvalues and eigenvectors
    # For n=100, this is fast. 
    # We want the eigenvector for the 2nd smallest eigenvalue.
    # np.linalg.eigh returns eigenvalues in ascending order.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvalue is at index 1
        # The corresponding eigenvector is eigenvectors[:, 1]
        fiedler_vec = eigenvectors[:, 1]
        # Assign labels based on the sign of the Fiedler vector
        # Usually 0 for negative, 1 for positive
        labels = (fiedler_vec > 0).astype(np.int8)
        return labels
    except Exception:
        # Fallback to random if spectral fails
        return np.random.randint(0, 2, size=n)

def _greedy_1_flip(adj, labels, deadline):
    """
    Greedy local search: flip a single vertex if it improves the cut.
    Returns the improved labels.
    """
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            # Calculate gain for flipping vertex v
            # If v is in group 0, flipping to 1 changes cut by:
            # (neighbors in 1) - (neighbors in 0)
            # If v is in group 1, flipping to 0 changes cut by:
            # (neighbors in 0) - (neighbors in 1)
            
            # Vectorized count of neighbors in group 0 and 1
            # adj[v] is the row of the adjacency matrix
            # labels == 0 gives a boolean mask for group 0
            # We need the sum of adj[v] where labels are 0
            # And sum of adj[v] where labels are 1
            
            # To be efficient, we can precompute or just do it inline.
            # For n=100, inline is fine.
            
            if labels[v] == 0:
                # Gain = #neighbors in 1 - #neighbors in 0
                # #neighbors in 0 = sum(adj[v][labels==0])
                # #neighbors in 1 = sum(adj[v][labels==1])
                n0 = adj[v][labels == 0].sum()
                n1 = adj[v][labels == 1].sum()
                gain = n1 - n0
            else:
                # Gain = #neighbors in 0 - #neighbors in 1
                n0 = adj[v][labels == 0].sum()
                n1 = adj[v][labels == 1].sum()
                gain = n0 - n1
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels

def _greedy_2_swap(adj, labels, deadline, max_passes=10):
    """
    Local search allowing swaps of two vertices.
    This helps escape local minima of 1-flip search.
    """
    n = len(labels)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue # Swapping two vertices in the same group does nothing
                
                # Calculate gain of swapping i and j
                # Let's simulate the swap
                # Before:
                # i is in group gi, j is in group gj
                # After:
                # i is in group gj, j is in group gi
                
                # The change in cut size can be calculated by looking at edges involving i or j.
                # Edges not involving i or j are unchanged.
                
                # Let's just compute the cut size change directly for efficiency?
                # No, computing full cut size is O(n^2). We are in an O(n^2) loop, so O(n^4) total.
                # For n=100, 10^8 ops might be too slow if we do many passes.
                # Let's optimize the gain calculation.
                
                # Gain = (new_cut - old_cut)
                # Let's derive the formula.
                # Let S0 be set of vertices in group 0, S1 in group 1.
                # Cut = sum_{u in S0, v in S1} A[u,v]
                
                # If we swap i (in S0) and j (in S1):
                # i moves to S1, j moves to S0.
                
                # Edges affected:
                # 1. Edge (i, j): 
                #    Before: i in S0, j in S1 -> Crosses. Count = 1 (if A[i,j]=1)
                #    After: i in S1, j in S0 -> Crosses. Count = 1 (if A[i,j]=1)
                #    Change = 0.
                
                # 2. Edges (i, k) where k != j:
                #    Before: i in S0. If k in S1, crosses. If k in S0, doesn't.
                #    After: i in S1. If k in S1, doesn't. If k in S0, crosses.
                #    Change for edge (i,k):
                #    If k in S1: Before Cross (1), After No (0) -> -1
                #    If k in S0: Before No (0), After Cross (1) -> +1
                #    So change is A[i,k] * (1 if k in S0 else -1)
                
                # 3. Edges (j, k) where k != i:
                #    Before: j in S1. If k in S0, crosses. If k in S1, doesn't.
                #    After: j in S0. If k in S0, doesn't. If k in S1, crosses.
                #    Change for edge (j,k):
                #    If k in S0: Before Cross (1), After No (0) -> -1
                #    If k in S1: Before No (0), After Cross (1) -> +1
                #    So change is A[j,k] * (-1 if k in S0 else 1)
                
                # Total Gain = Sum_{k in S0, k!=j} A[i,k] - Sum_{k in S1, k!=i} A[i,k] 
                #            + Sum_{k in S1, k!=i} A[j,k] - Sum_{k in S0, k!=j} A[j,k]
                
                # Note: k can be i or j in these sums? 
                # In sum for i, k!=j. In sum for j, k!=i.
                # Let's handle the i,j edge separately? We said change is 0.
                # Let's just sum over all k and adjust for i,j if needed.
                
                # Let's define:
                # N0(i) = neighbors of i in S0
                # N1(i) = neighbors of i in S1
                # N0(j) = neighbors of j in S0
                # N1(j) = neighbors of j in S1
                
                # Current state: i in S0, j in S1.
                # N0(i) includes j? No, j is in S1. So N0(i) does not include j.
                # N1(i) includes j? Yes.
                # N0(j) includes i? Yes.
                # N1(j) does not include i.
                
                # Gain terms:
                # From i moving to S1:
                # Gains edges to S0 (except j, but j is moving to S0 too, so let's be careful).
                # Let's use the explicit sums derived above.
                
                # Sum1 = sum_{k in S0, k!=j} A[i,k]
                # Sum2 = sum_{k in S1, k!=i} A[i,k]
                # Sum3 = sum_{k in S1, k!=i} A[j,k]
                # Sum4 = sum_{k in S0, k!=j} A[j,k]
                
                # Gain = Sum1 - Sum2 + Sum3 - Sum4
                
                # We can compute these using vectorization.
                # Mask S0 = (labels == 0)
                # Mask S1 = (labels == 1)
                
                # Sum1: adj[i] @ (S0 & (indices != j))
                # Sum2: adj[i] @ (S1 & (indices != i)) -> S1 doesn't contain i anyway since i in S0.
                # So Sum2 = adj[i] @ S1. But we must exclude j? 
                # In Sum2, k is in S1. j is in S1. So k can be j.
                # The term A[i,j] is included in Sum2.
                # In the derivation: "If k in S1: Before Cross (1)...".
                # For k=j: Before i in S0, j in S1 -> Cross. After i in S1, j in S0 -> Cross.
                # Change 0.
                # Formula Sum2 includes A[i,j]. 
                # Formula Sum3 includes A[j,i] (since i in S1? No, i in S0). 
                # Wait, Sum3 is sum_{k in S1} A[j,k]. i is in S0, so i is not in S1. So A[j,i] is NOT in Sum3.
                # Sum4 is sum_{k in S0} A[j,k]. j is in S1, so j is not in S0. i is in S0. So A[j,i] IS in Sum4.
                
                # Let's re-verify the "Change 0" for edge (i,j).
                # Sum1: k in S0, k!=j. i in S0. j in S1. So k!=j is naturally satisfied if we just sum over S0? 
                # No, if we sum adj[i] @ S0, does it include j? No, j is in S1. So Sum1 is just adj[i] @ S0.
                # Sum2: sum_{k in S1, k!=i} A[i,k]. i in S0, so k!=i is natural. Sum2 is adj[i] @ S1.
                # Sum3: sum_{k in S1, k!=i} A[j,k]. i in S0, so k!=i is natural. Sum3 is adj[j] @ S1.
                # Sum4: sum_{k in S0, k!=j} A[j,k]. j in S1, so k!=j is natural. Sum4 is adj[j] @ S0.
                
                # So Gain = (adj[i] @ S0) - (adj[i] @ S1) + (adj[j] @ S1) - (adj[j] @ S0)
                
                # Let's check the (i,j) edge contribution in this formula.
                # adj[i] @ S0: j is in S1, so 0.
                # adj[i] @ S1: j is in S1, so A[i,j].
                # adj[j] @ S1: i is in S0, so 0.
                # adj[j] @ S0: i is in S0, so A[j,i].
                # Total from (i,j) edge in formula: -A[i,j] - A[j,i] = -2*A[i,j].
                
                # But we established the change for edge (i,j) is 0.
                # Where is the discrepancy?
                # My derivation:
                # Change for (i,k) where k in S1: -1. (Includes k=j? Yes. So -A[i,j])
                # Change for (i,k) where k in S0: +1.
                # Change for (j,k) where k in S0: -1. (Includes k=i? Yes. So -A[j,i])
                # Change for (j,k) where k in S1: +1.
                
                # So the formula Gain = Sum1 - Sum2 + Sum3 - Sum4 correctly captures -A[i,j] and -A[j,i].
                # Why did I think change is 0?
                # Before: i in 0, j in 1. Edge crosses.
                # After: i in 1, j in 0. Edge crosses.
                # Change is 0.
                
                # Let's re-evaluate the sums.
                # Sum2 = sum_{k in S1} A[i,k]. This includes k=j. Value A[i,j]. Term is -Sum2. So -A[i,j].
                # Sum4 = sum_{k in S0} A[j,k]. This includes k=i. Value A[j,i]. Term is -Sum4. So -A[j,i].
                
                # Is there a +A[i,j] somewhere?
                # Sum1 = sum_{k in S0} A[i,k]. k=j is not in S0.
                # Sum3 = sum_{k in S1} A[j,k]. k=i is not in S1.
                
                # So the formula predicts a change of -2*A[i,j].
                # But the actual change is 0.
                # The error is in the "Change for (i,k)" logic.
                
                # Let's look at (i,j) specifically in the "After" state.
                # i is in S1, j is in S0.
                # Edge (i,j) crosses.
                
                # In the "Change" calculation for i moving to S1:
                # We said: "If k in S1: Before Cross (1), After No (0) -> -1".
                # For k=j: Before i in S0, j in S1 -> Cross. After i in S1, j in S0 -> Cross.
                # So for k=j, the change is 0, not -1.
                # The rule "If k in S1: -1" assumes k stays in S1. But j moves to S0.
                
                # So we must exclude j from the "k in S1" sum for i, and exclude i from the "k in S0" sum for j.
                
                # Corrected Sums:
                # Sum1 = sum_{k in S0, k!=j} A[i,k]. (j not in S0 anyway, so just adj[i]@S0)
                # Sum2 = sum_{k in S1, k!=i, k!=j} A[i,k]. (i not in S1, so just adj[i]@S1 - A[i,j])
                # Sum3 = sum_{k in S1, k!=i, k!=j} A[j,k]. (i not in S1, so just adj[j]@S1 - A[j,j]? No, A[j,j]=0. 
                #         But we need to exclude j? j is in S1. A[j,j]=0. So adj[j]@S1 is fine? 
                #         Wait, does j interact with itself? No. 
                #         Does the "After" state of j (in S0) affect edge (j,j)? No.
                #         But we are summing A[j,k]. k=j is in S1. A[j,j]=0.
                #         However, the logic "If k in S1: +1" for j moving to S0.
                #         If k=j, Before j in S1, j in S1 (self loop? no). 
                #         Let's stick to k != i, j.
                
                # Let's just compute the gain by explicitly calculating the cut size difference for the affected edges, or simpler:
                # Compute cut size before and after for the subgraph induced by i, j and their neighbors?
                # Too complex.
                
                # Let's use the corrected formula:
                # Gain = (adj[i] @ S0) - (adj[i] @ S1 - A[i,j]) + (adj[j] @ S1 - A[j,i]) - (adj[j] @ S0)
                # Since A[i,j] = A[j,i]:
                # Gain = (adj[i] @ S0) - (adj[i] @ S1) + A[i,j] + (adj[j] @ S1) - (adj[j] @ S0) - A[i,j]
                # Gain = (adj[i] @ S0) - (adj[i] @ S1) + (adj[j] @ S1) - (adj[j] @ S0)
                
                # Wait, the -A[i,j] and +A[i,j] cancel out?
                # Let's re-verify the "Change 0" for (i,j).
                # Formula: - (adj[i] @ S1) includes -A[i,j].
                # Formula: + (adj[j] @ S1) ... j is in S1, i is in S0. 
                # Does adj[j] @ S1 include A[j,i]? No, i is in S0.
                # So where is the +A[i,j] to cancel the -A[i,j]?
                
                # Let's look at Sum3 again: sum_{k in S1} A[j,k].
                # In the "After" state, j is in S0.
                # Edge (j, k) for k in S1.
                # Before: j in S1, k in S1 -> No Cross.
                # After: j in S0, k in S1 -> Cross.
                # Change: +1.
                # This applies to all k in S1, EXCEPT if k is j? No, k!=j.
                # Does it apply to k=i? i is in S0. So i is not in S1.
                # So Sum3 does NOT include i.
                
                # So the term for edge (j,i) is NOT in Sum3.
                # Is it in Sum4?
                # Sum4 = sum_{k in S0} A[j,k].
                # Before: j in S1, k in S0 -> Cross.
                # After: j in S0, k in S0 -> No Cross.
                # Change: -1.
                # This applies to all k in S0, EXCEPT if k is j? j is in S1, so k!=j is natural.
                # Does it apply to k=i? i is in S0. Yes.
                # So Sum4 includes A[j,i].
                # So we have -A[j,i] from Sum4.
                
                # And from Sum2 (edges from i to S1):
                # Sum2 = sum_{k in S1} A[i,k].
                # Before: i in S0, k in S1 -> Cross.
                # After: i in S1, k in S1 -> No Cross.
                # Change: -1.
                # Applies to all k in S1. Includes k=j? Yes.
                # So we have -A[i,j] from Sum2.
                
                # So total change for (i,j) edge is -A[i,j] - A[j,i] = -2A[i,j].
                # But actual change is 0.
                
                # Why?
                # Because when i moves to S1 and j moves to S0, the edge (i,j) STILL crosses.
                # The "Change" logic for i: "After i in S1, k in S1 -> No Cross".
                # For k=j, j is NOT in S1 in the "After" state. j is in S0.
                # So for k=j, the condition "k in S1" is FALSE in the "After" state?
                # No, the "Change" is calculated based on the OLD state of other vertices?
                # No, local search usually assumes other vertices stay fixed.
                # But here we are swapping TWO vertices. So the "Other" vertices set changes for i and j.
                
                # This is getting complicated to derive analytically on the fly.
                # Given n=100, we can just compute the cut size difference directly?
                # Cut size is O(n^2). 
                # If we do this for every pair (i,j), it's O(n^4).
                # 100^4 = 10^8. 
                # In Python, 10^8 operations might take ~10-50 seconds.
                # We have 150s.
                # We can afford a few passes of 2-swap with direct cut calculation.
                
                # Let's try that. It's safer.
                
                # To speed up direct cut calculation, we can use the vectorized _cut_size.
                # But calling it 10^4 times (n^2/2) per pass might be slow due to function call overhead and memory allocation.
                # Let's try to optimize the gain calculation one more time.
                
                # Let's use the property:
                # Cut(S) = 1/2 * sum_{u,v} A[u,v] * (1 - label[u]*label[v]) ... no, labels are 0/1.
                # Let's stick to the code.
                
                # Alternative: Use the formula derived in literature for 2-swap.
                # Gain(i,j) = 2 * (A[i,j] - sum_{k in S0, k!=i,j} A[i,k] - sum_{k in S1, k!=i,j} A[j,k] ... )
                # Actually, let's just use the direct computation if it's fast enough.
                
                # Let's estimate speed of _cut_size.
                # mask = (labels[:, None] != labels[None, :]) & adj
                # This creates a 100x100 boolean array.
                # mask.sum()
                # This is very fast in NumPy.
                # 10,000 calls * 100x100 ops = 10^8 ops.
                # NumPy can do 10^8-10^9 simple ops per second.
                # So one pass might take ~0.1-1 second.
                # We can afford ~100 passes.
                
                # Let's implement direct calculation for safety.
                
                old_cut = _cut_size(adj, labels)
                
                # Swap i and j
                labels[i], labels[j] = labels[j], labels[i]
                new_cut = _cut_size(adj, labels)
                
                if new_cut > old_cut:
                    improved = True
                    # Keep the swap
                else:
                    # Revert
                    labels[i], labels[j] = labels[j], labels[i]
                    
        if not improved:
            break
    return labels

def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    labels = _spectral_init(adj)
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    
    # 2. Greedy 1-Flip Improvement
    if time.time() < deadline:
        labels = _greedy_1_flip(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # 3. 2-Swap Local Search
    # We try a few restarts or just one deep 2-swap search.
    # 2-swap is slower, so we limit it.
    if time.time() < deadline:
        labels = best_labels.copy()
        labels = _greedy_2_swap(adj, labels, deadline, max_passes=50)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        # If time remains, try a random restart with 1-flip
        if time.time() < deadline - 10:
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                labels = np.random.randint(0, 2, size=n)
                labels = _greedy_1_flip(adj, labels, deadline - 2)
                cut = _cut_size(adj, labels)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
# EVOLVE-BLOCK-END
