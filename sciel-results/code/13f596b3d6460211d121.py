import random
import time
import sys
from collections import deque

N = 12
MAX_SETS = 1 << N

def compute_threshold(trans):
    """
    trans: list of two lists, each of length N.
    trans[0][i] = destination of state i on letter 0
    trans[1][i] = destination of state i on letter 1
    Returns the length of the shortest reset word, or float('inf') if not synchronizing.
    """
    # Precompute images of all subsets for both letters
    img0 = [0] * MAX_SETS
    img1 = [0] * MAX_SETS
    
    # Precompute destination masks
    dest0 = [0] * N
    dest1 = [0] * N
    for i in range(N):
        dest0[i] = 1 << trans[0][i]
        dest1[i] = 1 << trans[1][i]
        
    # Compute images for all subsets
    # Optimization: Use bitwise operations where possible, but loop is fine for N=12
    for S in range(MAX_SETS):
        m0 = 0
        m1 = 0
        # Iterate through set bits of S
        # Since N is small, a simple loop is fast enough
        for i in range(N):
            if S & (1 << i):
                m0 |= dest0[i]
                m1 |= dest1[i]
        img0[S] = m0
        img1[S] = m1
        
    # BFS on subset automaton
    full_set = (1 << N) - 1
    visited = [False] * MAX_SETS
    visited[full_set] = True
    
    queue = deque()
    queue.append((full_set, 0))
    
    while queue:
        current_set, dist = queue.popleft()
        
        # Check if singleton
        if current_set & (current_set - 1) == 0:
            return dist
            
        # Expand
        next0 = img0[current_set]
        if not visited[next0]:
            visited[next0] = True
            queue.append((next0, dist + 1))
            
        next1 = img1[current_set]
        if not visited[next1]:
            visited[next1] = True
            queue.append((next1, dist + 1))
            
    return float('inf')

def local_search(trans, time_limit):
    """
    Perform greedy local search starting from trans.
    Returns the best trans found.
    """
    current_trans = [row[:] for row in trans]
    current_threshold = compute_threshold(current_trans)
    
    if current_threshold == float('inf'):
        return current_trans, 0
        
    improved = True
    while improved:
        improved = False
        # Try changing every single transition
        for a in range(2):
            for q in range(N):
                old_dest = current_trans[a][q]
                # Try all possible new destinations
                for new_dest in range(N):
                    if new_dest == old_dest:
                        continue
                    current_trans[a][q] = new_dest
                    new_threshold = compute_threshold(current_trans)
                    
                    if new_threshold != float('inf') and new_threshold > current_threshold:
                        current_threshold = new_threshold
                        improved = True
                        # Continue searching from this new state
                        # We don't break out of loops immediately to see if more improvements are possible
                        # But typically we accept the first improvement and restart the scan
                        # To be more efficient, we can break and restart the whole scan
                        break
                if improved:
                    break
            if improved:
                break
                
        # Check time
        if time.time() > time_limit:
            break
            
    return current_trans, current_threshold

def generate_random_sync_trans():
    """
    Generate a random synchronizing automaton.
    Strategy: Start with a cycle and add random perturbations, ensuring synchronizability.
    """
    # Start with a cycle: 0->1->2...->N-1->0 for letter 0
    # Letter 1: all go to 0 (reset)
    # This is synchronizing with threshold N-1.
    # We will perturb this.
    
    trans = [[0]*N for _ in range(2)]
    
    # Base structure: Cycle for 0, Reset for 1
    for i in range(N):
        trans[0][i] = (i + 1) % N
        trans[1][i] = 0
        
    # Perturb some transitions to potentially increase threshold
    # We want to break the simple reset structure to force longer synchronization
    # Randomly change destinations for letter 1
    for i in range(N):
        if random.random() < 0.5:
            trans[1][i] = random.randint(0, N-1)
            
    # Ensure it's still synchronizing? 
    # If not, we might need to repair or just let local search handle it (it will reject non-sync)
    # But local search requires a starting sync point to evaluate.
    # Let's check if it's sync. If not, revert to a known sync one.
    if compute_threshold(trans) == float('inf'):
        # Fallback to a simple sync automaton
        for i in range(N):
            trans[0][i] = (i + 1) % N
            trans[1][i] = 0
            # Randomly break one link in the cycle for letter 0 to create a "pinwheel" effect?
            # Actually, just returning the base is safe.
            pass
            
    return trans

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin for 60s limit
    
    best_trans = None
    best_threshold = 0
    
    random.seed(42)
    
    # We will perform multiple restarts of local search
    # Each restart starts from a different random initialization
    
    while time.time() - start_time < time_limit:
        # Generate a random starting point
        # To get diverse starting points, we can vary the perturbation level
        start_trans = generate_random_sync_trans()
        
        # Run local search
        current_trans, current_threshold = local_search(start_trans, time_limit)
        
        if current_threshold > best_threshold:
            best_threshold = current_threshold
            best_trans = [row[:] for row in current_trans]
            
        # If we found a very high threshold, we might stop early or continue refining
        # Theoretical max is ~121. If we hit 100+, we are doing well.
        
        # Small sleep to allow time check to be effective in tight loops? No, time check is inside local_search.
        
    if best_trans is None:
        # Fallback if no sync automaton was found (should not happen)
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_trans

if __name__ == "__main__":
    res = construct()
    print(f"Best Threshold: {compute_threshold(res)}")
