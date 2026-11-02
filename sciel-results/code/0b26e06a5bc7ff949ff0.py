import numpy as np

N = 100

def construct():
    # Initialize random binary sequence with sum N/2
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Shuffle
    np.random.shuffle(h)
    
    best_h = h.copy()
    best_cost = float('inf')
    
    # Helper to get autocorrelation
    def get_cost(seq):
        # Circular autocorrelation
        f = np.fft.fft(seq)
        power = f * np.conj(f)
        ac = np.real(np.fft.ifft(power))
        # ac[0] is sum of squares = N/2
        # We want max of ac[1:]
        return np.max(ac[1:])
    
    current_cost = get_cost(h)
    best_cost = current_cost
    
    # Simple Local Search / Simulated Annealing
    # Since N is small, we can do many iterations
    for i in range(50000):
        # Pick random index with 1 and random index with 0
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        idx1 = np.random.choice(ones)
        idx0 = np.random.choice(zeros)
        
        # Swap
        h[idx1] = 0
        h[idx0] = 1
        
        new_cost = get_cost(h)
        
        # Accept if better
        if new_cost < best_cost:
            best_cost = new_cost
            best_h = h.copy()
        else:
            # Revert
            h[idx1] = 1
            h[idx0] = 0
            
    return best_h
