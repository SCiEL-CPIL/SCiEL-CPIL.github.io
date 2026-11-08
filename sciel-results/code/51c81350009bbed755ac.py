import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 180.0  # 180s limit, keep margin
    
    # Coefficients for H0, H4, H8
    # c12 is derived from g(0)=0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # c0 - 3c4 + 105c8 - 10395c12 = 0
    
    def get_c12(c0, c4, c8):
        return (c0 - 3*c4 + 105*c8) / 10395.0

    def wigner_max(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        coeffs = [c0, c4, c8, c12]
        
        # Grid for x and p
        # The Wigner function of a Gaussian times Hermite is a polynomial in x,p times a Gaussian.
        # W(x,p) = (1/2pi) ∫ g(x+t)g(x-t) exp(-2ipt) dt
        # For g(x) = P(x) exp(-x^2/2), W(x,p) is a polynomial in x,p times exp(-x^2-p^2).
        
        # We can compute the Wigner function analytically or numerically.
        # Numerical integration is safer.
        
        x = np.linspace(-10, 10, 400)
        p = np.linspace(-10, 10, 400)
        X, P = np.meshgrid(x, p)
        
        # Compute g(x)
        # Using scipy.special.hermite? Or explicit polynomials?
        # H0=1, H2=x^2-1, H4=x^4-6x^2+3, H6=x^6-15x^3+45x-15? No.
        # Probabilists vs Physicists.
        # Standard Hermite polynomials for QM (Physicists):
        # H0=1
        # H1=2x
        # H2=4x^2-2
        # H4=16x^4-48x^2+12
        # H8=256x^8-3072x^6+11520x^4-13440x^2+336
        # H12=4096x^12 - 98304x^10 + 1182720x^8 - 6720000x^6 + 19353600x^4 - 29030400x^2 + 1330560
        
        # Let's use scipy.special.hermite to be safe, or just numpy.poly1d.
        # scipy.special.hermite(n) returns the polynomial.
        
        def g_func(x_vals):
            h0 = np.ones_like(x_vals)
            h4 = 16*x_vals**4 - 48*x_vals**2 + 12
            h8 = 256*x_vals**8 - 3072*x_vals**6 + 11520*x_vals**4 - 13440*x_vals**2 + 336
            h12 = 4096*x_vals**12 - 98304*x_vals**10 + 1182720*x_vals**8 - 6720000*x_vals**6 + 19353600*x_vals**4 - 29030400*x_vals**2 + 1330560
            return (c0*h0 + c4*h4 + c8*h8 + c12*h12) * np.exp(-x_vals**2/2)
        
        # Wigner Function via FFT or direct integration
        # W(x,p) = (1/2pi) ∫ g(x+t)g(x-t) e^{-2ipt} dt
        # Let t = y/2. dt = dy/2.
        # W(x,p) = (1/4pi) ∫ g(x+y/2)g(x-y/2) e^{-ipy} dy
        
        # We can compute this for all x on the grid.
        # To speed up, we can use the fact that g is a sum of Hermite-Gaussians.
        # The Wigner function of a linear combination is a sum of cross terms.
        # W_{H_k, H_m} are known.
        
        # However, numerical integration on the grid is more robust for arbitrary coeffs.
        # Let's sample t.
        t = np.linspace(-10, 10, 500)
        dt = t[1]-t[0]
        
        W = np.zeros_like(X)
        
        # Vectorized computation might be too memory heavy for 400x400x500.
        # Let's reduce grid size for speed. 200x200x200.
        x = np.linspace(-8, 8, 200)
        p = np.linspace(-8, 8, 200)
        X, P = np.meshgrid(x, p)
        t = np.linspace(-8, 8, 200)
        dt = t[1]-t[0]
        
        # Precompute g at shifted points?
        # g(x+t) and g(x-t) for all x, t.
        # This is O(Nx^2 * Nt). 200^2 * 200 = 8e6. Manageable.
        
        # Let's just iterate over t for simplicity and speed in Python if vectorized properly.
        # Actually, let's use the analytic form of the Wigner function for Hermite functions.
        # W_{n,m}(x,p) = (-1)^{min(n,m)} 2^{n+m} / (2pi) sqrt(n!m!) / (2^{max(n,m)} max(n,m)!) ... this is getting complicated.
        
        # Let's stick to numerical integration but optimize it.
        # We can compute the integral for each x using FFT if we fix x.
        # But x varies.
        
        # Alternative: Use the property that the Wigner function of a Gaussian is a Gaussian.
        # The Wigner function of H_k(x) e^{-x^2/2} is a polynomial in x, p times e^{-x^2-p^2}.
        # Let's compute the coefficients of the polynomial in x, p.
        
        # This is too complex to derive on the fly.
        
        # Let's use a coarser grid and simple trapezoidal rule.
        # Time limit is 180s.
        
        def compute_wigner_grid(c0, c4, c8, c12):
            x = np.linspace(-6, 6, 100)
            p = np.linspace(-6, 6, 100)
            t = np.linspace(-6, 6, 200)
            dt = t[1]-t[0]
            
            X, P = np.meshgrid(x, p)
            W = np.zeros_like(X)
            
            # g(x)
            def g_val(z):
                h0 = 1.0
                h4 = 16*z**4 - 48*z**2 + 12
                h8 = 256*z**8 - 3072*z**6 + 11520*z**4 - 13440*z**2 + 336
                h12 = 4096*z**12 - 98304*z**10 + 1182720*z**8 - 6720000*z**6 + 19353600*z**4 - 29030400*z**2 + 1330560
                return (c0*h0 + c4*h4 + c8*h8 + c12*h12) * np.exp(-z**2/2)
            
            # For each x, compute the integral over t
            # W(x,p) = (1/2pi) ∫ g(x+t)g(x-t) e^{-2ipt} dt
            # This is essentially the Fourier transform of the function h_x(t) = g(x+t)g(x-t) at frequency 2p.
            
            # We can compute this using FFT for each x.
            # t is uniform.
            # Let N_t = len(t).
            # We need g(x+t) and g(x-t).
            
            # To speed up, we can precompute g on a fine grid and interpolate?
            # Or just compute directly.
            
            # Let's try to vectorize over x.
            # X is (Nx, Np). t is (Nt,).
            # We need g(X[:,None] + t[None,:]) and g(X[:,None] - t[None,:]).
            # This creates arrays of size (Nx, Nt).
            # Nx=100, Nt=200. 20k points. Very fast.
            
            xt_plus = X[:, None] + t[None, :]
            xt_minus = X[:, None] - t[None, :]
            
            g_plus = g_val(xt_plus)
            g_minus = g_val(xt_minus)
            
            h_x_t = g_plus * g_minus
            
            # Now we need the Fourier transform of h_x_t with respect to t, evaluated at 2p.
            # The integral is ∫ h_x_t(t) e^{-i 2p t} dt.
            # This is the Fourier transform at frequency 2p.
            # We can use np.fft.fft if we shift appropriately, but p is arbitrary.
            # Since p is on a grid, we can use the discrete transform if the grid matches.
            # t grid: -6 to 6, 200 points. dt = 12/199.
            # p grid: -6 to 6, 100 points.
            # The frequency range of the FFT is [-1/(2dt), 1/(2dt)].
            # 1/(2dt) = 199/24 ≈ 8.3.
            # p goes up to 6, so 2p goes up to 12. This is outside the Nyquist limit of the t-grid!
            # We need a larger t-grid or smaller p-grid.
            
            # Let's increase Nt to 400 and decrease p range or increase t range.
            # If t is -10 to 10, Nt=400, dt=20/399≈0.05. Nyquist freq = 1/(2*0.05)=10.
            # 2p max = 12. Still not enough.
            
            # Let's use a larger t grid. Nt=800, range -10 to 10. dt=0.025. Nyquist=20.
            # 2p max = 12. OK.
            
            # Let's re-define grids inside the function to be safe.
            x = np.linspace(-5, 5, 80)
            p = np.linspace(-5, 5, 80)
            t = np.linspace(-10, 10, 800)
            dt = t[1]-t[0]
            
            X, P = np.meshgrid(x, p)
            
            xt_plus = X[:, None] + t[None, :]
            xt_minus = X[:, None] - t[None, :]
            
            g_plus = g_val(xt_plus)
            g_minus = g_val(xt_minus)
            h_x_t = g_plus * g_minus
            
            # Compute Fourier Transform
            # We want ∫ h(t) e^{-i 2p t} dt.
            # np.fft.fft computes ∑ h_k e^{-i 2π k n / N}.
            # Frequency k corresponds to 2π k / (N dt).
            # We want frequency ω = 2p.
            # k = ω N dt / (2π) = 2p N dt / (2π) = p N dt / π.
            
            Nt = len(t)
            k_indices = np.round(P * Nt * dt / np.pi)
            # k_indices can be negative or > Nt. We need to handle wrapping or use ifftshift.
            # The FFT assumes input is in order 0..N-1.
            # Our t is centered at 0. We should ifftshift h_x_t before FFT.
            
            h_shifted = np.fft.ifftshift(h_x_t, axes=1)
            H_fft = np.fft.fft(h_shifted, axis=1)
            
            # The FFT result H_fft[k] corresponds to frequency k.
            # We need the value at k_indices.
            # Since k_indices might be out of bounds, we can use interpolation or just take the max.
            # Actually, we just need the MAX of the Wigner function.
            # W(x,p) = (1/2π) * (1/dt) * H_fft[k] * dt? No.
            # ∫ h(t) e^{-iωt} dt ≈ dt ∑ h(t_k) e^{-iωt_k}.
            # This is dt * (Unnormalized FFT).
            # So W(x,p) ≈ (1/2π) * dt * H_fft[k].
            
            # We want to find the max of W over x,p.
            # Instead of mapping every p to a k, we can just find the max of the entire FFT spectrum for each x?
            # No, the Wigner function is defined for all p. The max over p is the max of the Fourier transform magnitude?
            # W(x,p) is real. The FT of a real even function (h_x(t) is even) is real and even.
            # So we can just take the real part.
            
            # Let's find the max over the whole grid of (1/2π) * dt * real(H_fft).
            # This gives an upper bound on the max W(x,p) for the specific p-grid used in FFT?
            # No, the FFT samples specific frequencies. The true max might be between samples.
            # But for a smooth function, it's a good approximation.
            
            W_approx = (1.0 / (2.0 * np.pi)) * dt * np.real(H_fft)
            
            # We want the max over x and the FFT frequency index.
            # The FFT frequency index k corresponds to p_k = π k / (Nt dt).
            # We don't care about the specific p, just the max value.
            
            max_w = np.max(W_approx)
            
            return max_w

    def objective(c):
        c0, c4, c8 = c
        if time.time() - start_time > deadline - 5:
            return 1e10
        return compute_wigner_grid(c0, c4, c8, get_c12(c0, c4, c8))

    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Try a few random starts to avoid local minima
    best_x = x0
    best_val = objective(x0)
    
    for i in range(5):
        if time.time() - start_time > deadline - 10:
            break
            
        # Random perturbation
        x_rand = x0 + np.random.normal(0, 0.1, 3)
        
        res = opt.minimize(objective, x_rand, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    
    # Final polish with SLSQP if time permits
    if time.time() - start_time < deadline - 20:
        res = opt.minimize(objective, best_x, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    return np.array(best_x)
