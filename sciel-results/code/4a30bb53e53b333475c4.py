delta = sum_{k≠i,j} (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) 
      + sum_{k≠i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
      + (a[i,j]-a[j,j])*(b[v,u]-b[v,v]) + (a[j,i]-a[i,i])*(b[u,v]-b[u,u])
      + (a[i,i]-a[j,i])*(b[v,v]-b[u,v]) + (a[j,j]-a[i,j])*(b[u,u]-b[v,u])
