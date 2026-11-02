"""
Strategy: Neural Network Parameterization for Autoconvolution Minimization.

The step function $h(x)$ is parameterized by a small latent vector $z$ passed through
a deep feedforward neural network with ReLU activations. The output is squashed to be
non-negative (via softmax or exp) and normalized. We optimize $z$ using gradient descent
with backpropagation to minimize the normalized maximum of the autoconvolution.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 600
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_score_numpy(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

class AutoConvNet(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=[64, 128, 256]):
        super(AutoConvNet, self).__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, N))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        # Output raw logits, we will apply softmax/exp later
        return self.net(z)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialize best result with a simple heuristic (uniform or cosine)
    # Uniform score is 1/N = 1/600 ~ 0.00166
    # Cosine window (1+cos(4*pi*x))/2 on [-0.25, 0.25]
    x = np.linspace(-0.25, 0.25, N)
    h_cosine = 0.5 * (1 + np.cos(4 * np.pi * x))
    h_cosine = np.maximum(0, h_cosine)
    best_h = h_cosine / np.sum(h_cosine)
    best_score = compute_score_numpy(best_h)
    
    # Also try uniform
    h_uni = np.ones(N) / N
    score_uni = compute_score_numpy(h_uni)
    if score_uni < best_score:
        best_score = score_uni
        best_h = h_uni

    # 2. Setup Neural Network Optimizer
    latent_dim = 32
    model = AutoConvNet(latent_dim=latent_dim, hidden_dims=[64, 128, 256]).to(DEVICE)
    z = torch.randn(latent_dim, device=DEVICE) * 0.1
    z.requires_grad = True
    
    optimizer = torch.optim.Adam([z], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Pre-compute FFT of ones? No, just use standard conv.
    # We will compute the score in PyTorch for gradients.
    
    def torch_score(z_val, model):
        # Generate h
        logits = model(z_val)
        # Use softmax to ensure positive and normalized? 
        # Or exp + normalize? 
        # Softmax is bounded (0,1) and sums to 1. 
        # But we want non-negative step heights. 
        # The problem asks for "sequence of nonnegative step heights". 
        # Normalization is part of the score formula: max(conv) / (sum(h))^2.
        # Since the score is scale-invariant (numerator scales with h^2, denominator with h^2),
        # we can just normalize h to sum=1 for numerical stability during optimization,
        # or let the network learn the scale. 
        # Let's use Softmax for stability, which forces sum=1 and h>0.
        h = F.softmax(logits, dim=0)
        
        # Compute autoconvolution
        # Linear convolution of h with h
        # h has size N. conv has size 2N-1.
        conv = F.conv1d(h.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0).flip(1), mode='valid')
        # Wait, F.conv1d with mode='valid' gives size N. 
        # We need 'full' equivalent.
        # Let's use FFT for speed and ease of full convolution.
        h_fft = torch.fft.rfft(h, n=4*N)
        conv_fft = h_fft * h_fft
        conv = torch.fft.irfft(conv_fft, n=4*N)[:2*N-1]
        
        # Score: max(conv) / (sum(h))^2
        # Since h is softmax, sum(h)=1. So score = max(conv).
        # But wait, the original score is max(conv) / (sum h)^2.
        # If we use softmax, sum h = 1. So score = max(conv).
        # Is this equivalent? 
        # Original: h. Score = max(h*h) / (sum h)^2.
        # If we scale h by c, max(c^2 h*h) / (c sum h)^2 = c^2 max(h*h) / c^2 (sum h)^2. Invariant.
        # So yes, normalizing to sum=1 is valid.
        
        max_conv = torch.max(conv)
        return max_conv, h

    # Training loop
    best_torch_score = best_score
    patience_counter = 0
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        optimizer.zero_grad()
        score, h_torch = torch_score(z, model)
        loss = score
        
        if not torch.isfinite(loss):
            # Reset z if loss explodes
            z.data = torch.randn(latent_dim, device=DEVICE) * 0.1
            continue
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([z], 1.0)
        
        optimizer.step()
        scheduler.step(loss)
        
        current_score = loss.item()
        
        if current_score < best_score:
            best_score = current_score
            # Extract numpy h
            with torch.no_grad():
                h_np = h_torch.cpu().numpy()
            # Ensure non-negative and finite
            h_np = np.maximum(0, h_np)
            if np.sum(h_np) > 0:
                best_h = h_np / np.sum(h_np)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 50:
                break
                
        # Periodic check of raw score to ensure consistency
        if step % 100 == 0:
            raw_score = compute_score_numpy(best_h)
            if raw_score < best_score:
                best_score = raw_score
            # Occasionally try to perturb z to escape local minima
            if step % 500 == 0 and step > 0:
                z.data += torch.randn_like(z.data) * 0.1

    # 3. Final polish with local coordinate descent on the best NN output
    h = best_h.copy()
    # Normalize h to sum 1 for stability in local search
    h = h / np.sum(h)
    
    # Simple local refinement: try to flatten the peak of the convolution
    # The peak is likely at lag 0 (autocorrelation) or near it.
    # We can try to reduce the weights at the center if the peak is at 0.
    
    for _ in range(20):
        if time.time() > deadline:
            break
        conv = np.convolve(h, h, mode='full')
        max_idx = np.argmax(conv)
        # If max is at 0 (center), we might want to reduce the center values of h
        # If max is elsewhere, it's more complex.
        
        # Perform a small gradient step on h directly (not latent)
        # Score = max(conv) / (sum h)^2
        # Approximate gradient of max(conv) w.r.t h
        # d(max(conv))/d(h[i]) is non-zero only for indices contributing to the max lag.
        # Let's just do a random perturbation + annealing
        temp = 0.01 * (1 - _/20.0)
        h_new = h + np.random.randn(N) * temp
        h_new = np.maximum(0, h_new)
        if np.sum(h_new) > 0:
            h_new = h_new / np.sum(h_new)
            new_score = compute_score_numpy(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new
                h = h_new

    return best_h.tolist()
