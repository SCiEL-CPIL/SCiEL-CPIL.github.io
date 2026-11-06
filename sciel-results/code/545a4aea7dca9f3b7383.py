import numpy as np
import time

N = 17

def construct():
    # 1. Generate Seed Network (Odd-Even Transposition)
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
    
    # 2. Prepare Verification Data
    all_inputs = (np.arange(2**N)[:, None] >> np.arange(N)[None, :]) & 1
    
    def verify(net):
        state = all_inputs.copy()
        for i, j in net:
            a = state[:, i]
            b = state[:, j]
            state[:, i] = np.minimum(a, b)
            state[:, j] = np.maximum(a, b)
        return np.all(np.diff(state, axis=1) >= 0)

    deadline = time.time() + 80.0 # 10s safety margin
    
    # Phase 1: Greedy 1-Removal
    while time.time() < deadline:
        changed = False
        for i in range(len(network)):
            if time.time() > deadline:
                break
            new_net = network[:i] + network[i+1:]
            if verify(new_net):
                network = new_net
                changed = True
                break
        if not changed:
            break
            
    # Phase 2: Greedy 2-Removal
    # Try to remove two comparators at once.
    # This helps escape local minima where no single removal is valid, 
    # but a pair removal is.
    
    # To avoid O(N^2) if N is large, we limit the scope or just do it if time permits.
    # N is ~100. N^2 is 10,000.
    # 10,000 verifications * 1ms = 10s.
    # We have ~80s left? No, we used some time in Phase 1.
    # Let's check time.
    
    if time.time() < deadline - 5.0: # Need at least 5s left
        while time.time() < deadline:
            changed = False
            n = len(network)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    # Remove i and j
                    # Note: if we remove i, the index of j shifts.
                    # Easier: create a set of indices to remove.
                    # Or just slice carefully.
                    # If i < j:
                    # new_net = network[:i] + network[i+1:j] + network[j+1:]
                    # If we are in the inner loop, j is relative to the original network?
                    # Yes, we iterate i, j over the current network length.
                    
                    new_net = network[:i] + network[i+1:j] + network[j+1:]
                    
                    if verify(new_net):
                        network = new_net
                        changed = True
                        break # Break inner loop, restart outer? 
                        # Restarting outer is safer to re-evaluate indices.
                
            if changed:
                continue
            else:
                break
                
    return network
