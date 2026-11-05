import numpy as np
from scipy.optimize import minimize

N = 400

def _objective(h):
    # Minimize L2 norm squared of autoconvolution
    # c = convolve(h, h)
    # return sum(c**2)
    # Using FFT for speed
    # Pad to 2N to avoid circular convolution
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = H * H
    # c is real
    c = np.fft.irfft(C, n=size)
    return float(np.dot(c, c))

def _jac(h):
    # Gradient is 2 * convolve(h, h, h)
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = H * H * H
    grad = 2.0 * np.fft.irfft(C, n=size)
    # The gradient with respect to h[0:N] is the first N elements?
    # Wait, the convolution is linear.
    # The gradient vector should be of length N.
    # The triple convolution $h*h*h$ has length 3N-2.
    # The gradient $\frac{\partial J}{\partial h_m} = 2 \sum_k c_k h_{k-m}$.
    # This is the cross-correlation of $c$ and $h$?
    # Actually, $\nabla_h \|h*h\|_2^2 = 2 h * h * h$.
    # The result of $h*h*h$ is a vector of length $3N-2$.
    # But $h$ has length $N$.
    # The gradient must have length $N$.
    # The formula $\sum_k c_k h_{k-m}$ sums over $k$.
    # $c$ has length $2N-1$. $h$ has length $N$.
    # The result is a vector of length $(2N-1) + N - 1 = 3N-2$?
    # No, the derivative with respect to $h_m$ is a scalar.
    # $\frac{\partial J}{\partial h_m} = 2 \sum_{k} c_k \frac{\partial c_k}{\partial h_m}$.
    # $\frac{\partial c_k}{\partial h_m} = h_{k-m}$ (if indices valid).
    # So $\frac{\partial J}{\partial h_m} = 2 \sum_k c_k h_{k-m}$.
    # This is the convolution of $c$ and $h$ evaluated at index $m$?
    # Let $g = c * h$. $g_m = \sum_k c_k h_{m-k}$.
    # We have $\sum_k c_k h_{k-m}$.
    # Since $c$ is symmetric ($c_k = c_{2N-2-k}$? No, $c$ is symmetric around its center),
    # and $h$ is real, the convolution is symmetric?
    # Actually, $c$ is symmetric: $c_k = c_{2N-2-k}$ is false.
    # $c_k = \sum_i h_i h_{k-i}$.
    # $c_{2N-2-k} = \sum_i h_i h_{2N-2-k-i}$. Let $j = 2N-2-k-i$. Then $i = 2N-2-k-j$.
    # This is getting confusing.
    # Let's just compute the gradient numerically or use the FFT property.
    # The gradient of $\|h*h\|_2^2$ is $2 (h*h*h)$.
    # But we only need the first N components?
    # No, the gradient is a vector of the same size as h.
    # The operation $h \to h*h*h$ maps $N \to 3N-2$.
    # But the functional is defined on $N$.
    # The gradient is the projection of the "full" gradient onto the support of h?
    # Yes.
    # So we compute $t = h*h*h$ (length 3N-2) and take the first N elements?
    # Or the center?
    # Let's check dimensions.
    # $h$ is $0..N-1$.
    # $h*h$ is $0..2N-2$.
    # $(h*h)*h$ is $0..3N-3$.
    # The derivative $\frac{\partial J}{\partial h_m}$ corresponds to the coefficient of $h_m$ in the expansion.
    # It is indeed the $m$-th element of the triple convolution?
    # Let's test with N=2.
    # h = [a, b].
    # c = [a^2, 2ab, b^2].
    # J = a^4 + 4a^2b^2 + b^4.
    # dJ/da = 4a^3 + 8ab^2 = 4a(a^2 + 2b^2).
    # Triple conv: h*h*h = [a, b] * [a^2, 2ab, b^2] * [a, b]?
    # h*h = [a^2, 2ab, b^2].
    # (h*h)*h = [a^3, 2a^2b, 2ab^2, b^3] * [a, b]? No.
    # [a^2, 2ab, b^2] * [a, b] = [a^3, 2a^2b + a^2b, 2ab^2 + b^2a, b^3] = [a^3, 3a^2b, 3ab^2, b^3].
    # First element is $a^3$.
    # dJ/da is $4a^3 + 8ab^2$.
    # This does not match $4 \times$ first element.
    # So the gradient is NOT simply the triple convolution.
    
    # Let's re-derive.
    # J = sum c_k^2.
    # dJ/dh_m = 2 sum c_k dc_k/dh_m.
    # c_k = sum_i h_i h_{k-i}.
    # dc_k/dh_m = h_{k-m} + h_{m} (if k-m != m) or 2h_m.
    # Basically, dc_k/dh_m = h_{k-m} + h_{k-m} (with index checks).
    # So dJ/dh_m = 2 sum_k c_k (h_{k-m} + h_{k-m})? No.
    # If we swap i and k-i, the term is the same.
    # So dc_k/dh_m = h_{k-m} (if m != k-m) + h_{k-m} (if m == k-m)?
    # Actually, $c_k = \sum_i h_i h_{k-i}$.
    # Term with $h_m$: $h_m h_{k-m} + h_{k-m} h_m = 2 h_m h_{k-m}$ if $m \ne k-m$.
    # If $m = k-m$, term is $h_m^2$, deriv is $2 h_m$.
    # So $dc_k/dh_m = 2 h_{k-m}$ (with $h_j=0$ outside range).
    # So $dJ/dh_m = 4 \sum_k c_k h_{k-m}$.
    # This is $4 (c * h^{-})_m$.
    # Since $c$ is symmetric, $c * h^{-} = c * h$?
    # $c_k = c_{2N-2-k}$? No.
    # $c_k$ is symmetric around $N-1$. $c_{N-1+j} = c_{N-1-j}$.
    # So $c$ is a symmetric sequence.
    # The convolution of a symmetric sequence with $h$ is not necessarily symmetric.
    # But we just need to compute it.
    
    # So Gradient = 4 * convolve(c, h) ?
    # Let's check N=2 again.
    # c = [a^2, 2ab, b^2].
    # conv(c, h) = [a^2, 2ab, b^2] * [a, b] = [a^3, 2a^2b + a^2b, 2ab^2 + b^2a, b^3] = [a^3, 3a^2b, 3ab^2, b^3].
    # dJ/da = 4a^3 + 8ab^2.
    # 4 * (conv(c,h))[0] = 4a^3. Mismatch.
    # Why?
    # $dc_0/dh_0 = 2a$. $c_0=a^2$. Term: $2 a^2 (2a) = 4a^3$.
    # $dc_1/dh_0 = 2b$. $c_1=2ab$. Term: $2 (2ab) (2b) = 8ab^2$.
    # $dc_2/dh_0 = 0$.
    # Sum: $4a^3 + 8ab^2$.
    # My formula $4 \sum c_k h_{k-m}$:
    # m=0.
    # k=0: $c_0 h_0 = a^2 a = a^3$.
    # k=1: $c_1 h_1 = 2ab b = 2ab^2$.
    # k=2: $c_2 h_2 = b^2 (0) = 0$.
    # Sum = $a^3 + 2ab^2$.
    # Times 4: $4a^3 + 8ab^2$. Matches!
    
    # So Gradient = 4 * (c * h) where the convolution is aligned such that index 0 corresponds to m=0?
    # The sum is $\sum_k c_k h_{k-m}$.
    # This is the cross-correlation of c and h?
    # $(c \star h)_m = \sum_k c_k h_{k-m}$.
    # So Gradient = 4 * (c \star h).
    
    # We can compute this using FFT.
    # Correlation is convolution with reversed h.
    # Or just use np.convolve and slice.
    
    pass
