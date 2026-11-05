import numpy as np

def priority(adjacency):
            # Ensure float for matrix ops
            adj = adjacency.astype(float)
            n = adj.shape[0]
            degrees = adj.sum(axis=1)
            
            # Avoid division by zero
            inv_degrees = 1.0 / (degrees + 1.0)
            
            # Sum of inverse degrees of neighbors
            neighbor_penalty = adj @ inv_degrees
            
            # Score: Lower degree is better (higher score).
            # Penalty for having many low-degree neighbors (which are valuable to keep).
            # We want to pick vertices that are NOT connected to many low-degree vertices.
            # So we subtract the penalty.
            # Base score: -degree
            # Penalty term: lambda * neighbor_penalty
            # Total: -degree - lambda * neighbor_penalty
            
            # Tuning lambda:
            # If lambda is too high, we might prioritize very isolated high-degree nodes?
            # Let's try lambda = 1.0 or 0.5.
            # Given the baseline is just -degree, adding a small contextual term should help.
            
            scores = -degrees - 0.5 * neighbor_penalty
            
            return scores
