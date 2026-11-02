import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

# Define Hermite polynomials explicitly for speed and precision
def H0(x): return np.ones_like(x)
def H4(x): return 16*x**4 - 48*x**2 + 12
def H8(x): return 256*x**8 - 4608*x**6 + 30720*x**4 - 61440*x**2 + 16800
def H12(x): return 4096*x**12 - 147456*x**10 + 2416512*x**8 - 19353600*x**6 + 62208000*x**4 - 74547840*x**2 + 10321920

# Pre-compute constants for c12 constraint
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 16800.0
H12_0 = 10321920.0

def get_c12(c0, c4, c8):
    return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

def objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = get_c12(c0, c4, c8)
    
    # Grid for integration
    # The functions decay as e^{-x^2/4}, so a range of [-10, 10] is sufficient
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    
    # g(x)
    g = (c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)) * np.exp(-x**2 / 4.0)
    
    # Fourier transform of g
    # The Fourier transform of H_n(x) e^{-x^2/4} is (i*sqrt(2*pi))^n * 2^{n/2} * H_n(sqrt(2*pi)*xi) * e^{-pi^2*xi^2} ?
    # Let's use the standard definition: F(xi) = int f(x) e^{-i 2 pi x xi} dx
    # For f(x) = e^{-x^2/4}, F(xi) = 2 sqrt(pi) e^{-4 pi^2 xi^2}.
    # This scaling is complicated. It is safer to use FFT on a grid if the grid is uniform and large enough,
    # or use the analytic property that the Fourier transform of a Hermite-Gaussian is another Hermite-Gaussian.
    # 
    # Property: If f(x) = H_n(x) e^{-x^2/4}, then \hat{f}(\xi) = C_n H_n(2\pi \xi) e^{-4\pi^2 \xi^2} ?
    # Actually, let's just use the fact that the L1 norm of the Fourier transform can be estimated.
    # However, the problem states "score is the upper bound on C4". 
    # The bound is typically A(f) A(\hat{f}).
    # For the test function, we want to minimize the ratio of the bound to the actual uncertainty, or just the bound itself?
    # The task says "reduce the upper bound on A(f)*A(Fourier(f))".
    # The upper bound is likely derived from the L1 norms of the test function and its dual.
    # In the context of the specific paper (Goncalves et al.), the bound is related to the L1 norms of g and \hat{g}.
    # Specifically, the constant C is bounded by ||g||_1 ||\hat{g}||_1.
    # So we minimize ||g||_1 ||\hat{g}||_1.
    
    # Let's compute ||g||_1
    norm_g = np.sum(np.abs(g)) * dx
    
    # Compute \hat{g}
    # We can use the analytic form.
    # \mathcal{F}[H_n(x) e^{-x^2/4}] = (i \sqrt{2\pi})^n 2^{n/2} H_n(2\pi \xi) e^{-4\pi^2 \xi^2} ? No, the scaling is tricky.
    # Let's use a numerical FFT approach on a sufficiently large grid for the Fourier transform.
    # To avoid aliasing, the grid needs to be large.
    # Alternatively, since g is a polynomial times Gaussian, \hat{g} is also a polynomial times Gaussian.
    # We can derive the coefficients of \hat{g} analytically.
    # \mathcal{F}[x^k e^{-x^2/4}] = (-i 2\pi)^k \frac{\partial^k}{\partial (2\pi \xi)^k} (2\sqrt{\pi} e^{-4\pi^2 \xi^2})
    # This is equivalent to: \hat{g}(\xi) = \sum c_k \hat{H}_k(\xi)
    # where \hat{H}_k(\xi) is the Fourier transform of H_k(x) e^{-x^2/4}.
    
    # Let's use the property: \mathcal{F}[H_n(x) e^{-x^2/2}] = i^n H_n(\xi) e^{-\xi^2/2} (for unitary FT).
    # Our FT is not unitary. \hat{f}(\xi) = \int f(x) e^{-i 2\pi x \xi} dx.
    # Let's stick to numerical integration of the Fourier transform using the analytic expression for the FT of the basis functions.
    # Basis: b_k(x) = H_k(x) e^{-x^2/4}.
    # \hat{b}_k(\xi) = (i \sqrt{2\pi})^k 2^{k/2} H_k(2\pi \xi) e^{-4\pi^2 \xi^2} ? 
    # Let's check k=0: H_0=1. \hat{1 \cdot e^{-x^2/4}} = 2\sqrt{\pi} e^{-4\pi^2 \xi^2}.
    # Formula: (i\sqrt{2\pi})^0 2^0 H_0(2\pi \xi) e^{-4\pi^2 \xi^2} = 2\sqrt{\pi} e^{-4\pi^2 \xi^2}. Matches.
    # Check k=1: H_1=2x. \mathcal{F}[2x e^{-x^2/4}] = 2 (-i 2\pi) \frac{\partial}{\partial (2\pi \xi)} (2\sqrt{\pi} e^{-4\pi^2 \xi^2}) ?
    # \frac{\partial}{\partial \xi} e^{-4\pi^2 \xi^2} = -8\pi^2 \xi e^{-4\pi^2 \xi^2}.
    # \mathcal{F}[x e^{-x^2/4}] = -i 2\pi (2\sqrt{\pi}) (-8\pi^2 \xi) e^{-4\pi^2 \xi^2} ? No.
    # \mathcal{F}[x f(x)] = \frac{i}{2\pi} \frac{d}{d\xi} \hat{f}(\xi).
    # \hat{e^{-x^2/4}} = 2\sqrt{\pi} e^{-4\pi^2 \xi^2}.
    # \frac{d}{d\xi} = 2\sqrt{\pi} (-8\pi^2 \xi) e^{-4\pi^2 \xi^2} = -16\pi^{5/2} \xi e^{-4\pi^2 \xi^2}.
    # \mathcal{F}[x e^{-x^2/4}] = \frac{i}{2\pi} (-16\pi^{5/2} \xi e^{-4\pi^2 \xi^2}) = -8i \pi^{3/2} \xi e^{-4\pi^2 \xi^2}.
    # \mathcal{F}[2x e^{-x^2/4}] = -16i \pi^{3/2} \xi e^{-4\pi^2 \xi^2}.
    # Formula: (i\sqrt{2\pi})^1 2^{1/2} H_1(2\pi \xi) e^{-4\pi^2 \xi^2} = i \sqrt{4\pi} (4\pi \xi) e^{-4\pi^2 \xi^2} = i 2\sqrt{\pi} 4\pi \xi e^{-4\pi^2 \xi^2} = 8i \pi^{3/2} \xi e^{-4\pi^2 \xi^2}.
    # Sign difference? H_1(x) = 2x. \mathcal{F}[2x e^{-x^2/4}] = -16i \pi^{3/2} \xi ...
    # The sign depends on the convention of H_n and FT.
    # Let's just use numerical FFT on a fine grid for the Fourier transform to be safe.
    # Grid for FT:
    N = 2048
    L = 40.0 # Total length [-20, 20]
    xi = np.linspace(-L/2, L/2, N, endpoint=False)
    dxi = L/N
    
    # We need g(xi) on this grid.
    g_xi = (c0 * H0(xi) + c4 * H4(xi) + c8 * H8(xi) + c12 * H12(xi)) * np.exp(-xi**2 / 4.0)
    # This is g evaluated at xi, which is NOT \hat{g}(\xi).
    # We need \hat{g}(\xi) = \int g(x) e^{-i 2\pi x \xi} dx.
    # Using FFT:
    # We have g(x) on a grid x.
    # \hat{g}(\xi) \approx \sum g(x_j) e^{-i 2\pi x_j \xi_k} \Delta x.
    # This is a DFT if the grids are dual.
    # Let's use scipy.fft.fftshift and fft.
    # It's easier to just use the analytic form with the correct coefficients.
    # Let's re-verify the analytic FT of H_n(x) e^{-x^2/4}.
    # \mathcal{F}[H_n(x) e^{-x^2/4}] = (-i 2\pi)^n \frac{\partial^n}{\partial (2\pi \xi)^n} (2\sqrt{\pi} e^{-4\pi^2 \xi^2}) ? No.
    # \mathcal{F}[H_n(x) e^{-x^2/2}] = i^n H_n(\xi) e^{-\xi^2/2} (unitary).
    # Let's use the unitary FT result and convert.
    # Let U be unitary FT. \hat{f}_{std}(\xi) = \sqrt{2\pi} (\mathcal{F}_{std} f)(\xi) ? No.
    # \mathcal{F}_{std} f(\xi) = \int f(x) e^{-i 2\pi x \xi} dx.
    # U f(\xi) = \frac{1}{\sqrt{2\pi}} \int f(x) e^{-i x \xi} dx.
    # So \mathcal{F}_{std} f(\xi) = \sqrt{2\pi} U f(\xi / (2\pi)) ? No.
    # \mathcal{F}_{std} f(\xi) = \int f(x) e^{-i (2\pi \xi) x} dx. Let \omega = 2\pi \xi.
    # \mathcal{F}_{std} f(\xi) = \sqrt{2\pi} (U f)(\omega) ? No, U f(\omega) = \frac{1}{\sqrt{2\pi}} \int f(x) e^{-i \omega x} dx.
    # So \int f(x) e^{-i \omega x} dx = \sqrt{2\pi} U f(\omega).
    # Thus \mathcal{F}_{std} f(\xi) = \sqrt{2\pi} U f(2\pi \xi).
    
    # We know U [H_n(x) e^{-x^2/2}] = i^n H_n(\xi) e^{-\xi^2/2}.
    # Our function is g(x) = \sum c_k H_k(x) e^{-x^2/4}.
    # Let y = x/\sqrt{2}. Then x = y\sqrt{2}, dx = \sqrt{2} dy.
    # e^{-x^2/4} = e^{-y^2/2}.
    # H_k(x) = H_k(y\sqrt{2}).
    # This scaling makes the Hermite argument different.
    # It is much simpler to just use numerical integration for the FT.
    
    # Let's use a large grid for x and compute the FT via FFT.
    N_x = 4096
    L_x = 40.0
    x_grid = np.linspace(-L_x/2, L_x/2, N_x, endpoint=False)
    dx = L_x / N_x
    
    # Evaluate g on x_grid
    g_vals = (c0 * H0(x_grid) + c4 * H4(x_grid) + c8 * H8(x_grid) + c12 * H12(x_grid)) * np.exp(-x_grid**2 / 4.0)
    
    # Compute FT
    # \hat{g}(\xi) \approx \sum g(x_j) e^{-i 2\pi x_j \xi_k} dx
    # This is equivalent to dx * FFT(g_vals) if the frequencies are aligned.
    # The FFT computes \sum g_j e^{-i 2\pi j k / N}.
    # x_j = -L/2 + j dx. \xi_k = k / L.
    # e^{-i 2\pi x_j \xi_k} = e^{-i 2\pi (-L/2 + j dx) k/L} = e^{i \pi k} e^{-i 2\pi j k / N}.
    # So \hat{g}(\xi_k) \approx dx \cdot (-1)^k \cdot \text{FFT}(g_vals)_k.
    # We need to shift the FFT output to center the frequency.
    G_fft = np.fft.fft(g_vals)
    G_fft = np.fft.fftshift(G_fft)
    # The frequencies are xi_k = (k - N/2) / L.
    # The phase factor e^{i \pi k} depends on the original index.
    # Let's just use the property that the FT of a real even function is real and even.
    # H0, H4, H8, H12 are even. So g(x) is even. \hat{g}(\xi) is real and even.
    # So we can just take the magnitude of the FFT? No, the FFT of an even function is real.
    # But the shift introduces a phase.
    # Let's use a simpler numerical integration for the FT since the grid is small enough.
    # Or use the analytic form derived from the unitary FT property with scaling.
    
    # Let's use the analytic form:
    # \mathcal{F}_{std} [H_n(x) e^{-x^2/4}] (\xi)
    # Let f(x) = H_n(x) e^{-x^2/4}.
    # We found \mathcal{F}_{std} [e^{-x^2/4}] = 2\sqrt{\pi} e^{-4\pi^2 \xi^2}.
    # \mathcal{F}_{std} [x^m e^{-x^2/4}] = (-i 2\pi)^m \frac{\partial^m}{\partial (2\pi \xi)^m} (2\sqrt{\pi} e^{-4\pi^2 \xi^2}) ?
    # No, \mathcal{F}[x^m f] = \frac{i^m}{(2\pi)^m} \frac{d^m}{d\xi^m} \hat{f}.
    # \mathcal{F}[x^m e^{-x^2/4}] = \frac{i^m}{(2\pi)^m} \frac{d^m}{d\xi^m} (2\sqrt{\pi} e^{-4\pi^2 \xi^2}).
    # Since H_n(x) is a polynomial in x, we can expand it, take derivatives, and sum.
    # This is computationally feasible for n=12.
    
    # However, for the purpose of optimization, we can just use the L1 norm of g and the L1 norm of \hat{g}.
    # We can approximate \|\hat{g}\|_1 by numerically integrating |\hat{g}(\xi)|.
    # Let's use the FFT approach carefully.
    # g is even. \hat{g} is even and real.
    # \hat{g}(\xi) = 2 \int_0^\infty g(x) \cos(2\pi x \xi) dx.
    # We can use np.trapz on the cosine transform.
    # Grid for x: 0 to 20.
    x_pos = np.linspace(0, 20, 2000)
    dx_pos = x_pos[1] - x_pos[0]
    g_pos = (c0 * H0(x_pos) + c4 * H4(x_pos) + c8 * H8(x_pos) + c12 * H12(x_pos)) * np.exp(-x_pos**2 / 4.0)
    
    # Grid for xi: 0 to 5.
    xi_pos = np.linspace(0, 5, 2000)
    dxi_pos = xi_pos[1] - xi_pos[0]
    
    # Compute \hat{g}(\xi) = 2 * sum g(x) cos(2 pi x xi) dx
    # This is an outer product, which is expensive (2000x2000).
    # 4 million operations per evaluation. With 1000 evaluations, 4e9 ops. Might be too slow in Python.
    # We need a faster method.
    
    # Let's use the analytic FT coefficients.
    # \hat{g}(\xi) = \sum_{k \in \{0,4,8,12\}} c_k \hat{H}_k(\xi)
    # where \hat{H}_k(\xi) = \mathcal{F}[H_k(x) e^{-x^2/4}](\xi).
    # We can pre-compute \hat{H}_k(\xi) on a fixed grid.
    
    # Let's define a function to compute \hat{H}_k(\xi) analytically.
    # \mathcal{F}[x^m e^{-x^2/4}] = \frac{i^m}{(2\pi)^m} \frac{d^m}{d\xi^m} (2\sqrt{\pi} e^{-4\pi^2 \xi^2}).
    # We can compute the derivatives of the Gaussian using Hermite polynomials again!
    # \frac{d^m}{d\xi^m} e^{-a \xi^2} = (-1)^m (2a)^{m/2} H_m(\sqrt{a} \xi) e^{-a \xi^2} ?
    # Yes, \frac{d^n}{dx^n} e^{-x^2} = (-1)^n H_n(x) e^{-x^2}.
    # Here a = 4\pi^2. \sqrt{a} = 2\pi.
    # \frac{d^m}{d\xi^m} e^{-4\pi^2 \xi^2} = (-1)^m (8\pi^2)^{m/2} H_m(2\pi \xi) e^{-4\pi^2 \xi^2}.
    # So \mathcal{F}[x^m e^{-x^2/4}] = \frac{i^m}{(2\pi)^m} 2\sqrt{\pi} (-1)^m (8\pi^2)^{m/2} H_m(2\pi \xi) e^{-4\pi^2 \xi^2}.
    # Simplify:
    # = 2\sqrt{\pi} i^m (-1)^m (8\pi^2)^{m/2} / (2\pi)^m H_m(2\pi \xi) e^{-4\pi^2 \xi^2}
    # = 2\sqrt{\pi} (i \cdot -1)^m (8\pi^2 / 4\pi^2)^{m/2} H_m(2\pi \xi) e^{-4\pi^2 \xi^2}
    # = 2\sqrt{\pi} (-i)^m (2)^{m/2} H_m(2\pi \xi) e^{-4\pi^2 \xi^2}.
    
    # So if H_k(x) = \sum_{m} a_{k,m} x^m, then
    # \hat{H}_k(\xi) = \sum_{m} a_{k,m} 2\sqrt{\pi} (-i)^m 2^{m/2} H_m(2\pi \xi) e^{-4\pi^2 \xi^2}.
    # Since k is even, m is even. (-i)^m is real.
    # This allows us to compute \hat{H}_k(\xi) efficiently on a grid.
    
    # Let's pre-compute the coefficients a_{k,m} for H0, H4, H8, H12.
    # H0(x) = 1. a_{0,0} = 1.
    # H4(x) = 16x^4 - 48x^2 + 12. a_{4,4}=16, a_{4,2}=-48, a_{4,0}=12.
    # H8(x) = 256x^8 - 4608x^6 + 30720x^4 - 61440x^2 + 16800.
    # H12(x) = 4096x^{12} - 147456x^{10} + 2416512x^8 - 19353600x^6 + 62208000x^4 - 74547840x^2 + 10321920.
    
    # We will set up a grid for \xi and pre-compute the basis functions \phi_m(\xi) = 2\sqrt{\pi} (-i)^m 2^{m/2} H_m(2\pi \xi) e^{-4\pi^2 \xi^2}.
    # Then \hat{H}_k(\xi) = \sum a_{k,m} \phi_m(\xi).
    # And \hat{g}(\xi) = \sum c_k \hat{H}_k(\xi).
    
    # Grid for \xi
    N_xi = 1000
    xi_grid = np.linspace(-5, 5, N_xi)
    dxi = xi_grid[1] - xi_grid[0]
    
    # Pre-compute \phi_m for m in {0, 2, 4, 6, 8, 10, 12}
    phi = {}
    for m in range(0, 13, 2):
        # (-i)^m = (-1)^{m/2}
        sign = (-1)**(m//2)
        coeff = 2 * np.sqrt(np.pi) * sign * (2**(m/2.0))
        # H_m(2\pi \xi)
        H_m_xi = eval_hermite(m, 2 * np.pi * xi_grid)
        gauss = np.exp(-4 * np.pi**2 * xi_grid**2)
        phi[m] = coeff * H_m_xi * gauss
        
    # Pre-compute \hat{H}_k
    a_coeffs = {
        0: {0: 1.0},
        4: {4: 16.0, 2: -48.0, 0: 12.0},
        8: {8: 256.0, 6: -4608.0, 4: 30720.0, 2: -61440.0, 0: 16800.0},
        12: {12: 4096.0, 10: -147456.0, 8: 2416512.0, 6: -19353600.0, 4: 62208000.0, 2: -74547840.0, 0: 10321920.0}
    }
    
    hat_H = {}
    for k in [0, 4, 8, 12]:
        hat_H[k] = np.zeros_like(xi_grid)
        for m, a in a_coeffs[k].items():
            hat_H[k] += a * phi[m]
            
    # Now objective function can be fast
    def fast_objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # ||g||_1
        # Use a fixed grid for x
        x_grid_obj = np.linspace(-10, 10, 1000)
        dx_obj = x_grid_obj[1] - x_grid_obj[0]
        g_vals = (c0 * H0(x_grid_obj) + c4 * H4(x_grid_obj) + c8 * H8(x_grid_obj) + c12 * H12(x_grid_obj)) * np.exp(-x_grid_obj**2 / 4.0)
        norm_g = np.sum(np.abs(g_vals)) * dx_obj
        
        # ||\hat{g}||_1
        hat_g_vals = c0 * hat_H[0] + c4 * hat_H[4] + c8 * hat_H[8] + c12 * hat_H[12]
        norm_hat_g = np.sum(np.abs(hat_g_vals)) * dxi
        
        return norm_g * norm_hat_g

    # Optimization
    start_time = time.time()
    best_score = float('inf')
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Multi-start
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() - start_time > 150:
            break
            
        # Random initial guess
        if i == 0:
            x0 = best_coeffs
        else:
            x0 = np.random.normal(0, 0.5, 3)
            
        res = minimize(fast_objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12})
        
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
            
    return best_coeffs

# We need to define H0, H4, H8, H12 globally or inside construct?
# They are defined inside construct in the draft above. Let's move them out or define them properly.

def construct():
    import numpy as np
    import time
    from scipy.optimize import minimize
    from scipy.special import eval_hermite

    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 4608*x**6 + 30720*x**4 - 61440*x**2 + 16800
    def H12(x): return 4096*x**12 - 147456*x**10 + 2416512*x**8 - 19353600*x**6 + 62208000*x**4 - 74547840*x**2 + 10321920

    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 16800.0
    H12_0 = 10321920.0

    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    # Pre-compute FT basis
    N_xi = 1000
    xi_grid = np.linspace(-5, 5, N_xi)
    dxi = xi_grid[1] - xi_grid[0]
    
    phi = {}
    for m in range(0, 13, 2):
        sign = (-1)**(m//2)
        coeff = 2 * np.sqrt(np.pi) * sign * (2**(m/2.0))
        H_m_xi = eval_hermite(m, 2 * np.pi * xi_grid)
        gauss = np.exp(-4 * np.pi**2 * xi_grid**2)
        phi[m] = coeff * H_m_xi * gauss
        
    a_coeffs = {
        0: {0: 1.0},
        4: {4: 16.0, 2: -48.0, 0: 12.0},
        8: {8: 256.0, 6: -4608.0, 4: 30720.0, 2: -61440.0, 0: 16800.0},
        12: {12: 4096.0, 10: -147456.0, 8: 2416512.0, 6: -19353600.0, 4: 62208000.0, 2: -74547840.0, 0: 10321920.0}
    }
    
    hat_H = {}
    for k in [0, 4, 8, 12]:
        hat_H[k] = np.zeros_like(xi_grid)
        for m, a in a_coeffs[k].items():
            hat_H[k] += a * phi[m]
            
    def fast_objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        x_grid_obj = np.linspace(-10, 10, 1000)
        dx_obj = x_grid_obj[1] - x_grid_obj[0]
        g_vals = (c0 * H0(x_grid_obj) + c4 * H4(x_grid_obj) + c8 * H8(x_grid_obj) + c12 * H12(x_grid_obj)) * np.exp(-x_grid_obj**2 / 4.0)
        norm_g = np.sum(np.abs(g_vals)) * dx_obj
        
        hat_g_vals = c0 * hat_H[0] + c4 * hat_H[4] + c8 * hat_H[8] + c12 * hat_H[12]
        norm_hat_g = np.sum(np.abs(hat_g_vals)) * dxi
        
        return norm_g * norm_hat_g

    start_time = time.time()
    best_score = float('inf')
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Check initial
    init_score = fast_objective(best_coeffs)
    best_score = init_score
    
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - start_time > 170:
            break
            
        if i == 0:
            x0 = best_coeffs
        else:
            x0 = np.random.normal(0, 1.0, 3)
            
        try:
            res = minimize(fast_objective, x0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return best_coeffs
