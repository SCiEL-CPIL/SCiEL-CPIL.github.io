import time
import random
import numpy as np

def construct():
    n = 256
    start = time.time()
    deadline = start + 118.0

    # 1. Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d)
    # d >= 1, a+3d <= n
    # Indices are 0-based
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)
    if num_aps == 0:
        # Fallback if no APs (shouldn't happen for n=256)
        return [0] * n

    ap_array = np.array(aps, dtype=np.int32)
    
    # 2. Build the interaction matrix (Adjacency-like) for the quadratic form
    # We want to minimize the number of monochromatic APs.
    # Let x_i in {-1, 1} be the color.
    # An AP (i, j, k, l) is monochromatic if x_i=x_j=x_k=x_l.
    # This is equivalent to x_i x_j x_k x_l = 1.
    # We can decompose this into pairwise interactions if we fix a structure, 
    # but a direct 4-wise interaction is hard for standard QP solvers.
    
    # However, the prompt suggests a "reduced basis" and "linear dependencies".
    # A simpler structural change: Treat the problem as minimizing the sum of 
    # "disagreements" or use a different objective.
    
    # Let's stick to the subgradient idea on the continuous relaxation.
    # Objective: F(x) = sum_{AP} (1 - prod(x_a, x_b, x_c, x_d)) / 2 
    # This is 0 if mono, 1 if not. Minimizing this maximizes non-mono APs.
    # Actually, we want to MINIMIZE mono APs.
    # So we want to MAXIMIZE sum_{AP} (1 - prod(x_a, x_b, x_c, x_d)) / 2.
    # Or MINIMIZE sum_{AP} prod(x_a, x_b, x_c, x_d).
    
    # Let's define the energy E(x) = sum_{AP} x_a x_b x_c x_d.
    # If all same, term is 1. If mixed, term is -1 or 0 (if not all same, prod is -1? No).
    # If 3 same, 1 diff: prod = -1.
    # If 2 same, 2 diff: prod = 1.
    # So minimizing E(x) doesn't directly correspond to minimizing mono count.
    
    # Let's use the explicit count of mono APs.
    # Indicator(I_mono) = (1 + x_a x_b + x_a x_c + ... + x_a x_b x_c x_d) / 16 ? No.
    # A robust way to count mono APs in {-1,1} space:
    # Let S = x_a + x_b + x_c + x_d.
    # If all 1: S=4. If all -1: S=-4.
    # If mixed: |S| < 4.
    # We can use a function that is 1 when |S|=4 and 0 otherwise.
    # f(S) = (S^4 - 8S^2 + 16) / 16 ?
    # S=4: (256 - 128 + 16)/16 = 144/16 = 9? No.
    
    # Let's go back to the subgradient of the "Mono Count".
    # C(x) = sum_{AP} I(x_a=x_b=x_c=x_d).
    # This function is non-differentiable.
    # Subgradient of I(x_a=x_b=x_c=x_d) w.r.t x_a:
    # If x_a=x_b=x_c=x_d, gradient is undefined (kink). We can approximate.
    # If not, gradient is 0.
    
    # Alternative Strategy from Directive: "Reduced basis of independent variables".
    # The Thue-Morse sequence has a specific structure. 
    # Perhaps we can parameterize the coloring by a smaller set of variables?
    # For n=256, a direct parameterization is hard.
    
    # Let's implement the "Vectorized Subgradient with Momentum" on the 
    # continuous relaxation of the mono-count.
    # We define a smooth approximation of the mono-indicator.
    # Let g(v) = exp(-beta * v) where v is a measure of "disagreement".
    # Disagreement D = (x_a-x_b)^2 + (x_a-x_c)^2 + ...
    # If mono, D=0. If not, D>0.
    # We want to minimize sum exp(-beta * D).
    # This is differentiable!
    
    # D(x) for an AP (a,b,c,d) = (x_a-x_b)^2 + (x_a-x_c)^2 + (x_a-x_d)^2 + (x_b-x_c)^2 + (x_b-x_d)^2 + (x_c-x_d)^2
    # Note: (x_a-x_b)^2 = 2 if different, 0 if same.
    # Sum of pairwise squared diffs.
    # If all same: 0.
    # If 3-1: 3*2 + 3*0 = 6? 
    # Pairs: (3,1) has 3 pairs of (same) and 3 pairs of (diff). 
    # Wait, 4 elements. 
    # If 3 are 1, 1 is -1.
    # Pairs: 
    # 1-1: 3 pairs (0 diff)
    # 1--1: 3 pairs (2 diff)
    # Total D = 6.
    # If 2-2:
    # 1-1: 1 pair (0)
    # -1--1: 1 pair (0)
    # 1--1: 4 pairs (2 diff)
    # Total D = 8.
    
    # So D is 0, 6, or 8.
    # We want to minimize sum exp(-beta * D).
    # This will be minimized when D is large (non-mono).
    # Wait, we want to MINIMIZE mono count.
    # So we want to MAXIMIZE the number of non-mono APs.
    # Or MINIMIZE the sum of indicators of Mono APs.
    # I_mono = 1 if D=0, else 0.
    # Approximation: f(D) = exp(-beta * D) is NOT an indicator. 
    # f(0)=1, f(6)=exp(-6beta), f(8)=exp(-8beta).
    # If beta is large, f(6) and f(8) are close to 0.
    # So minimizing sum f(D) encourages D>0.
    # This works!
    
    beta = 2.0
    
    # Gradient of f(D) w.r.t x_i:
    # df/dD * dD/dx_i
    # f(D) = exp(-beta D)
    # df/dD = -beta exp(-beta D)
    # D = sum_{j<k} (x_j-x_k)^2
    # dD/dx_i = 2 * sum_{j != i} (x_i - x_j) = 2 * (3*x_i - sum_{j in AP, j!=i} x_j)
    
    # Let's set up the arrays.
    # We need to compute D and Gradient for all APs efficiently.
    
    # Precompute pairs for each AP? No, just use the 4 indices.
    # D_ap = (x_a-x_b)^2 + (x_a-x_c)^2 + (x_a-x_d)^2 + (x_b-x_c)^2 + (x_b-x_d)^2 + (x_c-x_d)^2
    
    # Initialize x in [-1, 1]
    # Start with Thue-Morse mapped to [-1, 1]
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.float64)
    x = 2 * tm - 1  # 0 -> -1, 1 -> 1
    
    # Momentum
    v = np.zeros(n)
    lr = 0.05
    momentum = 0.9
    
    best_x = x.copy()
    best_mono_count = np.inf
    
    def count_mono(x_int):
        # x_int is 0/1 or -1/1
        # Convert to 0/1 for easy counting if needed, or just check equality
        # Using -1/1:
        count = 0
        for idx in range(num_aps):
            a, b, c, d = ap_array[idx]
            if x_int[a] == x_int[b] == x_int[c] == x_int[d]:
                count += 1
        return count
    
    # Initial count
    x_int_initial = (x + 1) // 2
    best_mono_count = count_mono(x_int_initial)
    
    t = 0
    while time.time() < deadline - 1.0:
        # Compute D for all APs
        # xa, xb, xc, xd
        xa = x[ap_array[:, 0]]
        xb = x[ap_array[:, 1]]
        xc = x[ap_array[:, 2]]
        xd = x[ap_array[:, 3]]
        
        # D = (xa-xb)^2 + ...
        d_ab = xa - xb
        d_ac = xa - xc
        d_ad = xa - xd
        d_bc = xb - xc
        d_bd = xb - xd
        d_cd = xc - xd
        
        D = d_ab**2 + d_ac**2 + d_ad**2 + d_bc**2 + d_bd**2 + d_cd**2
        
        # Objective: Sum(exp(-beta * D))
        # We want to MINIMIZE this.
        
        # Gradient w.r.t x
        # grad_x = -beta * exp(-beta D) * dD/dx
        
        # dD/dx_i for each position in the AP
        # For position a (index 0 in AP):
        # dD/xa = 2 * (3*xa - xb - xc - xd)
        
        grad_a = 2 * (3 * xa - xb - xc - xd)
        grad_b = 2 * (3 * xb - xa - xc - xd)
        grad_c = 2 * (3 * xc - xa - xb - xd)
        grad_d = 2 * (xd * 3 - xa - xb - xc) # Wait, 3*xd - xa - xb - xc
        
        # Weight = -beta * exp(-beta D)
        weight = -beta * np.exp(-beta * D)
        
        # Accumulate gradients
        grad_x = np.zeros(n)
        
        # Use np.add.at for efficient accumulation
        np.add.at(grad_x, ap_array[:, 0], weight * grad_a)
        np.add.at(grad_x, ap_array[:, 1], weight * grad_b)
        np.add.at(grad_x, ap_array[:, 2], weight * grad_c)
        np.add.at(grad_x, ap_array[:, 3], weight * grad_d)
        
        # Update with Momentum
        v = momentum * v + lr * grad_x
        x = x - v
        
        # Clip to [-1, 1] to keep it in the hypercube
        x = np.clip(x, -1.0, 1.0)
        
        # Periodically check the discrete solution
        # Throwing in a random perturbation if stuck?
        # Let's check every 100 steps
        if t % 100 == 0:
            x_int = (x + 1) // 2
            curr_mono = count_mono(x_int)
            if curr_mono < best_mono_count:
                best_mono_count = curr_mono
                best_x = x.copy()
        
        # Adaptive Learning Rate
        # If we are not improving, reduce LR?
        # Or just keep it constant.
        
        t += 1
        
        # If time is running out, break
        if time.time() > deadline - 2.0:
            break
            
    # Final check
    x_int_final = (best_x + 1) // 2
    final_mono = count_mono(x_int_final)
    
    if final_mono < best_mono_count:
        best_mono_count = final_mono
        
    return x_int_final.tolist()
